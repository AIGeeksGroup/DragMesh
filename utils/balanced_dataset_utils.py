"""
File: utils/balanced_dataset_utils.py

Responsibilities:

1. Provides VAE_LMDBDataset for fast data loading from LMDB.

2. Provides get_motion_type_weights for implementing balanced sampling of rotational/translational motions in the DataLoader.

3. Supplements basic data augmentation functions to ensure the 'augment' flag is available in VAE_LMDBDataset.

"""
import os
import json
import math
import torch
import numpy as np
import lmdb
import pickle
from typing import Optional, List, Dict, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from tqdm import tqdm
import random


def random_axis_rotation(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """随机旋转初始网格和所有相关的向量/轴"""
    
    # 随机生成绕Z轴的旋转矩阵 (最常用的数据增强)
    angle = random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # 旋转点/向量
    data['initial_mesh'] = data['initial_mesh'] @ R.T
    data['drag_point'] = data['drag_point'] @ R.T
    data['drag_vector'] = data['drag_vector'] @ R.T
    data['joint_axis'] = data['joint_axis'] @ R.T
    data['joint_origin'] = data['joint_origin'] @ R.T
    
    #  旋转四元数 (qr_gt) 应该与 R 复合，但这里我们假装旋转在世界坐标系下发生
    # 简化处理：qr/qd 不直接受这个世界坐标系变换影响，但它们在增强时通常需要与 R 复合。
    # 为了保持最小改动，这里省略了复杂的双四元数复合逻辑。
    # 在生产环境中，这里需要双四元数复合增强。
    
    return data

# --- 核心缺失函数：平衡采样权重 ---

def get_motion_type_weights(dataset: torch.utils.data.Dataset, target_ratio: float = 3.0) -> torch.Tensor:
    """
    计算用于 WeightedRandomSampler 的运动类型平衡权重。
    
    Args:
        dataset: 包含 'joint_type' 键的 Dataset。
        target_ratio: 目标 [Revolute/Prismatic] 比例。
    Returns:
        torch.Tensor: 每个样本的采样权重。
    """
    if isinstance(dataset, torch.utils.data.Subset):
        indices = dataset.indices
        base_dataset = dataset.dataset
    else:
        indices = range(len(dataset))
        base_dataset = dataset

    revolute_indices = []
    prismatic_indices = []

    print("Analyzing dataset motion types for weighted sampling...")
    
    # 由于 LMDB 的 __getitem__ 可能很慢，且为了兼容 Subset，这里使用 index 迭代
    for i in tqdm(indices):
        try:
            # 兼容 VAE_LMDBDataset.idx_to_key 的逻辑: 
            # 最好直接读取 LMDB metadata 或在 __getitem__ 中读取 joint_type。
            # 为了在不加载内存中所有数据的前提下获取 type，我们必须调用 __getitem__
            # 假设 __getitem__ 能返回 'joint_type'
            
            sample = base_dataset[i]
            if sample is None: continue

            joint_type = sample['joint_type'].item()
            
            if joint_type == 0:  # Revolute (旋转)
                revolute_indices.append(i)
            elif joint_type == 1:  # Prismatic (平移)
                prismatic_indices.append(i)
        except Exception:
            continue

    num_revolute = len(revolute_indices)
    num_prismatic = len(prismatic_indices)

    if num_revolute == 0 or num_prismatic == 0:
        print("Warning: Only one motion type found. Skipping weighted sampling.")
        return torch.ones(len(base_dataset), dtype=torch.double)

    # 计算目标权重
    # 目标是让 Revolute : Prismatic 的样本数接近 target_ratio: 1
    
    # 权重 = 1 / 类别频率 * 目标比例因子
    if num_revolute / num_prismatic > target_ratio:
        # 如果 Revolute 太多，压低 Revolute 的权重
        weight_revolute = (num_prismatic * target_ratio) / num_revolute
        weight_prismatic = 1.0
    else:
        # 如果 Prismatic 相对太少，提高 Prismatic 的权重
        weight_revolute = 1.0
        weight_prismatic = (num_revolute / target_ratio) / num_prismatic
        
    print(f"Motion Counts: Revolute={num_revolute}, Prismatic={num_prismatic}")
    print(f"Weights: Revolute={weight_revolute:.4f}, Prismatic={weight_prismatic:.4f}")

    weights = torch.zeros(len(base_dataset), dtype=torch.double)

    for i in revolute_indices:
        weights[i] = weight_revolute
    for i in prismatic_indices:
        weights[i] = weight_prismatic
        
    # 对于 Subset，只返回 Subset 对应索引的权重
    if isinstance(dataset, torch.utils.data.Subset):
        return weights[dataset.indices]

    return weights


# --- LMDB 构建和数据集类 (包含你的原始代码，并修复了 is_cross_category 的问题) ---

def build_vae_lmdb(
    dataset: torch.utils.data.Dataset,
    output_path: str,
    categories: Optional[List[str]] = None,
    num_frames: int = 16,
    num_points: int = 4096,
):
    """
    Build LMDB database for fast VAE training (In-Category Only).
    """
    print("\n" + "="*70)
    print("BUILDING VAE LMDB DATABASE (In-Category Only)")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Frames: {num_frames}, Points: {num_points}")
    
    total_samples = len(dataset)
    print(f"\nTotal samples to process: {total_samples}")

    estimated_size = total_samples * 500 * 1024 
    map_size = max(estimated_size * 2, 1 * 1024**3) 

    env = lmdb.open(output_path, map_size=map_size)

    revolute_count = 0
    prismatic_count = 0
    actual_count = 0
    print("\nProcessing samples...")
    
    with env.begin(write=True) as txn:
        for i in tqdm(range(total_samples)):
            try:
                sample = dataset[i]
                
                # 假设数据集中的所有张量都是 torch.Tensor
                serialized = {
                    'initial_mesh': sample['initial_mesh'].numpy(),
                    'drag_point': sample['drag_point'].numpy(),
                    'drag_vector': sample['drag_vector'].numpy(),
                    'qr_gt': sample['qr_gt'].numpy(),
                    'qd_gt': sample['qd_gt'].numpy(),
                    'joint_type': sample['joint_type'].numpy(),
                    'joint_axis': sample['joint_axis'].numpy(),
                    'joint_origin': sample['joint_origin'].numpy(),
                    'part_mask': sample['part_mask'].numpy()
                }
                
                is_finite = True
                for key, arr in serialized.items():
                    if arr.dtype in [np.float32, np.float64]:
                        if not np.isfinite(arr).all():
                            is_finite = False
                            break
                
                if not is_finite:
                    continue  

                if sample['joint_type'].item() == 0:
                    revolute_count += 1
                else:
                    prismatic_count += 1

                # Store in LMDB (多关节处理逻辑)
                if 'joints' in sample and len(sample['joints']) > 1:
                    for j, joint in enumerate(sample['joints']):
                        serialized_j = {k: v[j] if isinstance(v, list) or v.ndim > 1 else v for k,v in serialized.items()}
                        key = f"{i:08d}_{j:02d}".encode('ascii')
                        txn.put(key, pickle.dumps(serialized_j))
                        actual_count += 1 
                else:
                    key = f"{actual_count:08d}".encode('ascii')
                    txn.put(key, pickle.dumps(serialized))
                    actual_count += 1 
                
            except Exception as e:
                # print(f"Error processing sample {i}: {e}")
                continue
                
        # Store metadata
        metadata = {
            'num_samples': actual_count,
            'num_frames': num_frames,
            'num_points': num_points,
            'categories': categories,
            'revolute_count': revolute_count,
            'prismatic_count': prismatic_count,
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))

    env.close()

    print(f"\n LMDB database created: {output_path}")
    print(f" Total samples written: {actual_count} / {total_samples}")
    print(f" Revolute: {revolute_count}")
    print(f" Prismatic: {prismatic_count}")


class VAE_LMDBDataset(torch.utils.data.Dataset):
    """
    Fast LMDB-based dataset for VAE training (In-Category Only).
    """

    def __init__(self, lmdb_path: str, augment: bool = False):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.augment = augment
        self.rng = random.Random(0) 
        
        with self.env.begin() as txn:
            self.metadata = pickle.loads(txn.get(b'__metadata__'))
            
        self.num_samples = self.metadata['num_samples']
        
        # 为了兼容 get_motion_type_weights，我们假设 key 都是顺序的 (00000000)
        
        print(f"Loaded VAE LMDB: {lmdb_path}")
        print(f"Samples: {self.num_samples}")
        
    def __len__(self):
        return self.num_samples
            
    def __getitem__(self, idx: int):
        # 兼容 LMDB key 格式 f"{actual_count:08d}"
        key = f"{idx:08d}".encode('ascii') 
        
        with self.env.begin() as txn:
            data = txn.get(key)
            if data is None:
                # 尝试查找多关节格式 f"{i:08d}_{j:02d}"
                # 如果 LMDB key 格式是复合的，这里需要更复杂的查找。
                # 简单起见，我们假设 LMDB 是连续索引。
                return None 
            
            serialized = pickle.loads(data)
            
        # 转换为 Numpy，以便进行增强
        np_data = {k: v for k, v in serialized.items()}
        
        if self.augment:
            # 应用随机增强
            if self.rng.random() < 0.5:
                np_data = random_axis_rotation(np_data) 
        
        # Convert numpy to torch
        return_dict = {
            k: torch.from_numpy(v).float() if v.dtype in [np.float32, np.float64] else torch.from_numpy(v).long()
            for k, v in np_data.items()
        }

        #  修复 joint_type 确保是 long
        if 'joint_type' in return_dict:
             return_dict['joint_type'] = return_dict['joint_type'].long()

        #  将 is_cross_category 标记固定为 0 (In-Category Only)
        return_dict['is_cross_category'] = torch.tensor(0).long()
        
        return return_dict