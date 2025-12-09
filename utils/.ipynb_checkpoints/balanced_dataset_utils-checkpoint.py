"""
Balanced dataset utilities for VAE training with category-based splits.

Key features:
1. Balanced sampling of revolute vs prismatic joints
2. Category-based train/val split (in-category and cross-category)
3. LMDB support for fast I/O
4. Motion type stratification
"""
import os
import json
import math
import torch
import numpy as np
import lmdb
import pickle
from typing import Optional, List, Dict, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.dataset_utils import FixedDragMeshDatasetV2, load_category_config
from modules.dual_quaternion import quaternion_mul, matrix_to_quaternion

def random_axis_rotation(angle_range_deg=180.0):
    angle = np.radians(np.random.uniform(-angle_range_deg, angle_range_deg))
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([[0, -axis[2], axis[1]], 
                  [axis[2], 0, -axis[0]], 
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    T = np.eye(4)
    T[:3, :3] = R
    return T

def analyze_dataset_motion_types(dataset_root: str, categories: List[str]) -> Dict[str, int]:
    """
    Analyze dataset to count revolute vs prismatic samples.

    Args:
        dataset_root: Root directory of dataset
        categories: List of categories to analyze

    Returns:
        Dictionary with 'revolute' and 'prismatic' counts
    """
    from utils.dataset_utils import FixedGAPartNetLoader

    loader = FixedGAPartNetLoader(dataset_root, categories=categories)

    revolute_count = 0
    prismatic_count = 0

    print(f"Analyzing motion types for {len(loader)} objects...")

    for i in range(len(loader)):
        try:
            sample = loader.generate_training_sample(i, num_frames=16)
            joint_type = sample['joint_type']

            if joint_type in ['revolute', 'continuous']:
                revolute_count += 1
            elif joint_type == 'prismatic':
                prismatic_count += 1
        except Exception as e:
            continue

    print(f"  Revolute: {revolute_count}")
    print(f"  Prismatic: {prismatic_count}")
    print(f"  Ratio (R:P): {revolute_count/max(prismatic_count, 1):.2f}:1")

    return {
        'revolute': revolute_count,
        'prismatic': prismatic_count,
        'total': revolute_count + prismatic_count
    }


def get_motion_type_weights(dataset: torch.utils.data.Dataset,
                            target_ratio: float = 3.0) -> torch.Tensor:
    """
    Compute sampling weights to balance revolute vs prismatic joints.

    Args:
        dataset: Dataset object
        target_ratio: Target ratio of revolute:prismatic (default 3:1)

    Returns:
        Tensor of sampling weights [num_samples]
    """
    print("Computing motion type weights for balanced sampling...")

    # Count motion types
    revolute_count = 0
    prismatic_count = 0
    motion_types = []

    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            joint_type = sample['joint_type'].item()

            if joint_type == 0:  # Revolute
                revolute_count += 1
                motion_types.append(0)
            else:  # Prismatic
                prismatic_count += 1
                motion_types.append(1)
        except Exception as e:
            motion_types.append(0)  # Default to revolute

    total_samples = len(dataset)
    print(f"  Revolute: {revolute_count} ({revolute_count/total_samples*100:.1f}%)")
    print(f"  Prismatic: {prismatic_count} ({prismatic_count/total_samples*100:.1f}%)")

    # Compute weights
    # Target: revolute_weight * revolute_count : prismatic_weight * prismatic_count = target_ratio : 1
    if prismatic_count == 0:
        print("  No prismatic samples, using uniform weights")
        return torch.ones(total_samples)

    # Weight for each class
    weight_revolute = 1.0
    weight_prismatic = (revolute_count / prismatic_count) / target_ratio

    print(f"  Target ratio: {target_ratio}:1")
    print(f"  Weight revolute: {weight_revolute:.3f}")
    print(f"  Weight prismatic: {weight_prismatic:.3f}")

    # Assign weights to each sample
    weights = []
    for motion_type in motion_types:
        if motion_type == 0:
            weights.append(weight_revolute)
        else:
            weights.append(weight_prismatic)

    return torch.tensor(weights, dtype=torch.float32)


def create_balanced_dataloaders(
    dataset_root: str,
    config_path: str,
    batch_size: int = 16,
    num_frames: int = 16,
    num_points: int = 4096,
    num_workers: int = 4,
    target_ratio: float = 3.0,
    val_split_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create balanced dataloaders with category-based split.

    Args:
        dataset_root: Root directory of dataset
        config_path: Path to category_split_v2.json
        batch_size: Batch size
        num_frames: Number of trajectory frames
        num_points: Number of points to sample
        num_workers: Number of data loading workers
        target_ratio: Target revolute:prismatic ratio
        val_split_ratio: Validation split ratio for in-category split

    Returns:
        (train_loader, val_in_loader, val_cross_loader)
    """
    print("\n" + "="*70)
    print("CREATING BALANCED DATALOADERS")
    print("="*70)

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_categories = config['train_categories']
    val_cross_categories = config['val_cross_category']['categories']

    print(f"\nDataset: {dataset_root}")
    print(f"Train categories: {len(train_categories)}")
    print(f"Val cross categories: {len(val_cross_categories)}")

    # Create train dataset (all training categories)
    print("\n1. Loading training dataset...")
    full_train_dataset = FixedDragMeshDatasetV2(
        dataset_root=dataset_root,
        num_frames=num_frames,
        num_points=num_points,
        categories=train_categories
    )
    print(f"   Total train samples: {len(full_train_dataset)}")

    # Split into train and val_in (stratified by motion type if possible)
    train_size = int((1 - val_split_ratio) * len(full_train_dataset))
    val_in_size = len(full_train_dataset) - train_size

    # Random split (can be improved with stratification)
    train_dataset, val_in_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_in_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train split: {len(train_dataset)} samples")
    print(f"   Val in-category split: {len(val_in_dataset)} samples")

    # Create val_cross dataset (unseen categories)
    print("\n2. Loading cross-category validation dataset...")
    val_cross_dataset = FixedDragMeshDatasetV2(
        dataset_root=dataset_root,
        num_frames=num_frames,
        num_points=num_points,
        categories=val_cross_categories
    )
    print(f"   Val cross-category samples: {len(val_cross_dataset)}")

    # Compute balanced sampling weights for training
    print("\n3. Computing balanced sampling weights...")
    train_weights = get_motion_type_weights(train_dataset, target_ratio=target_ratio)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    # Create dataloaders
    print("\n4. Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_in_loader = DataLoader(
        val_in_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    val_cross_loader = DataLoader(
        val_cross_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val in batches: {len(val_in_loader)}")
    print(f"   Val cross batches: {len(val_cross_loader)}")

    print("\n" + "="*70)
    print("DATALOADERS READY")
    print("="*70 + "\n")

    return train_loader, val_in_loader, val_cross_loader


def build_vae_lmdb(
    dataset: torch.utils.data.Dataset,
    output_path: str,
    categories: Optional[List[str]] = None,
    num_frames: int = 16,
    num_points: int = 4096,
    val_in_size: Optional[int] = None
):
    """
    Build LMDB database for fast VAE training.

    Pre-processes and stores all samples in LMDB for 10-100x faster loading.

    Args:
        dataset_root: Root directory of dataset
        output_path: Output path for LMDB database
        categories: Optional list of categories to include
        num_frames: Number of trajectory frames
        num_points: Number of points to sample
    """
    print("\n" + "="*70)
    print("BUILDING VAE LMDB DATABASE (from Dataset object)")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Categories: {len(categories) if categories else 'all'}")
    print(f"Frames: {num_frames}, Points: {num_points}")
    
    if val_in_size is not None:
        print(f"Validation In-Category Size: {val_in_size}")

    # Create dataset
    total_samples = len(dataset)
    print(f"\nTotal samples to process: {total_samples}")


    # Estimate LMDB size (each sample ~500KB)
    estimated_size = total_samples * 500 * 1024  # 500KB per sample
    map_size = max(estimated_size * 2, 1 * 1024**3)  # At least 1GB

    # Create LMDB
    env = lmdb.open(output_path, map_size=map_size)

    # Count motion types
    revolute_count = 0
    prismatic_count = 0
    
    actual_count = 0
    print("\nProcessing samples...")
    
    with env.begin(write=True) as txn:
        for i in range(total_samples):
            if i % 100 == 0:
                print(f"  Progress: {i}/{total_samples} ({i/total_samples*100:.1f}%)")

            try:
                sample = dataset[i]
                
                try:
                    obj_path = f"Index_{i}"
                except Exception:
                    obj_path = "Unknown Path"
                
                # Convert to numpy for efficient storage
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
                            print(f"\n  [Build LMDB] 警告: 样本 {i} (Obj: {obj_path})")
                            print(f"    在键 '{key}' 中包含 non-finite (Inf/NaN) 值。")
                            print(f"    该样本将被跳过，不会写入 LMDB。")
                            is_finite = False
                            break
                
                if not is_finite:
                    continue  

                # Count motion types
                if sample['joint_type'].item() == 0:
                    revolute_count += 1
                else:
                    prismatic_count += 1

                # Store in LMDB
                key = f"{actual_count:08d}".encode('ascii')
                txn.put(key, pickle.dumps(serialized))
                actual_count += 1 
                
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                print("\n" + "="*30 + " ERROR " + "="*30)
                print(f"  [Build LMDB] Error processing sample {i}")
                print(f"  Object Path: {obj_path}")
                print(f"  Error Type: {type(e).__name__}")
                print(f"  Error Details: {e}")
                print("="*67 + "\n")
                continue
                
        

        # Store metadata
        metadata = {
            'num_samples': actual_count,
            'num_frames': num_frames,
            'num_points': num_points,
            'categories': categories,
            'revolute_count': revolute_count,
            'prismatic_count': prismatic_count,
            'val_in_size': val_in_size
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))

    env.close()

    print(f"\n✅ LMDB database created: {output_path}")
    print(f"    Total samples written: {actual_count} / {total_samples}")
    print(f"   Revolute: {revolute_count}")
    print(f"   Prismatic: {prismatic_count}")
    print(f"   Size: {os.path.getsize(output_path) / 1024**2:.1f} MB")
    print("="*70 + "\n")
    
    data_file_path = os.path.join(output_path, 'data.mdb')
    try:
        if os.path.exists(data_file_path):
            file_size_mb = os.path.getsize(data_file_path) / 1024**2
            print(f"    Size: {file_size_mb:.1f} MB")
        else:
            print("    Size: (Could not find data.mdb for size calculation)")
    except Exception as e:
        print(f"    Size: (Error calculating size: {e})")
    
    print("="*70 + "\n")


class VAE_LMDBDataset(torch.utils.data.Dataset):
    """
    Fast LMDB-based dataset for VAE training.

    Usage:
        # 1. Build LMDB first:
        build_vae_lmdb(dataset_root, output_path, categories=TRAIN_CATEGORIES)

        # 2. Use in training:
        dataset = VAE_LMDBDataset(lmdb_path='dataset.lmdb')
    """

    def __init__(self, lmdb_path: str, augment: bool = False):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.augment = augment
        
        with self.env.begin() as txn:
            self.metadata = pickle.loads(txn.get(b'__metadata__'))
            
        self.num_samples = self.metadata['num_samples']
        
        self.val_in_size = self.metadata.get('val_in_size', None)
        
        print(f"Loaded VAE LMDB: {lmdb_path}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Revolute: {self.metadata['revolute_count']}")
        print(f"  Prismatic: {self.metadata['prismatic_count']}")
        print(f"  Augmentation: {self.augment}")
        
        if self.val_in_size is not None:
            num_cross = self.num_samples - self.val_in_size
            print(f"  Split: {self.val_in_size} In-Category samples, {num_cross} Cross-Category samples")
            
        
    def __len__(self):
        return self.num_samples
            
    def __getitem__(self, idx: int):
        key = f"{idx:08d}".encode('ascii')
        
        with self.env.begin() as txn:
            data = txn.get(key)
            if data is None:
                raise KeyError(f"Sample {idx} not found in LMDB")
            serialized = pickle.loads(data)
            
        if self.augment:
            T_rot = random_axis_rotation(180.0) # 绕随机轴旋转 -180 到 180 度
            R_rot = T_rot[:3, :3].astype(np.float32)
            
            serialized['initial_mesh'] = np.dot(serialized['initial_mesh'], R_rot.T)
            serialized['drag_point'] = np.dot(serialized['drag_point'], R_rot.T)
            serialized['joint_origin'] = np.dot(serialized['joint_origin'], R_rot.T)
            serialized['drag_vector'] = np.dot(serialized['drag_vector'], R_rot.T)
            serialized['joint_axis'] = np.dot(serialized['joint_axis'], R_rot.T)
            
            q_rot = matrix_to_quaternion(torch.from_numpy(R_rot)).numpy()
            qr_gt_orig = serialized['qr_gt']
            qd_gt_orig = serialized['qd_gt']
            
            q_rot_expanded = np.tile(q_rot, (qr_gt_orig.shape[0], 1))
                

            qr_gt_new = quaternion_mul(
                torch.from_numpy(q_rot_expanded), 
                torch.from_numpy(qr_gt_orig)
            ).numpy()
            
            qd_gt_new = quaternion_mul(
                torch.from_numpy(q_rot_expanded), 
                torch.from_numpy(qd_gt_orig)
            ).numpy()

            serialized['qr_gt'] = qr_gt_new
            serialized['qd_gt'] = qd_gt_new
            
            # 4. 添加 Jitter 噪声
            points = serialized['initial_mesh']
            noise = np.random.normal(0, 0.005, points.shape).astype(np.float32)
            serialized['initial_mesh'] = points + noise
    
                            
            # Convert numpy to torch
        return {
                'initial_mesh': torch.from_numpy(serialized['initial_mesh']).float(),
                'drag_point': torch.from_numpy(serialized['drag_point']).float(),
                'drag_vector': torch.from_numpy(serialized['drag_vector']).float(),
                'qr_gt': torch.from_numpy(serialized['qr_gt']).float(),
                'qd_gt': torch.from_numpy(serialized['qd_gt']).float(),
                'joint_type': torch.from_numpy(serialized['joint_type']).long(),
                'joint_axis': torch.from_numpy(serialized['joint_axis']).float(),
                'joint_origin': torch.from_numpy(serialized['joint_origin']).float(),
                'part_mask': torch.from_numpy(serialized['part_mask']).float()
        }
    
        is_cross_category = 0 # 默认: 0 (In-Category)
            if self.val_in_size is not None:
                # 如果 val_in_size 被设置了 (说明这是验证集)
                # 并且 idx 大于等于这个值，说明它是 Cross-Category
                if idx >= self.val_in_size:
                    is_cross_category = 1 # 1 = Cross-Category

            return_dict['is_cross_category'] = torch.tensor(is_cross_category).long()
    
    
if __name__ == "__main__":
    """Test balanced dataset utilities."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str,
                       default='H:/dragmesh/dataset/partnet_mobility_part')
    parser.add_argument('--config', type=str,
                       default='config/category_split_v2.json')
    parser.add_argument('--build_lmdb', action='store_true',
                       help='Build LMDB database')
    parser.add_argument('--lmdb_output', type=str,
                       default='dataset_vae.lmdb')
    args = parser.parse_args()

    if args.build_lmdb:
        # Build LMDB
        with open(args.config, 'r') as f:
            config = json.load(f)
        train_categories = config['train_categories']

        build_vae_lmdb(
            dataset_root=args.dataset_root,
            output_path=args.lmdb_output,
            categories=train_categories,
            num_frames=16,
            num_points=4096
        )
    else:
        # Test balanced dataloaders
        train_loader, val_in_loader, val_cross_loader = create_balanced_dataloaders(
            dataset_root=args.dataset_root,
            config_path=args.config,
            batch_size=4,
            num_frames=16,
            num_points=4096,
            num_workers=0,
            target_ratio=3.0
        )

        print("\nTesting dataloaders...")
        print("Train batch:")
        batch = next(iter(train_loader))
        for key, value in batch.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
