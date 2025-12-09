"""
Build LMDB database for fast VAE training (IN-CATEGORY VALIDATION ONLY).

This script reads the v2 config file and splits the data into:
1.  [prefix]_train.lmdb: Contains the training split (e.g., 90%)
    of ALL seen categories.
2.  [prefix]_val.lmdb: Contains the validation split (e.g., 10%)
    of ALL seen categories (In-Category only).

Usage:
    python scripts/build_lmdb.py \
        --dataset_root /path/to/partnet_mobility_part \
        --output_prefix /path/to/vae_data \
        --config config/category_split_v2_in_domain.json
"""
import sys
import os
import argparse
import json
import torch
from torch.utils.data import Subset, random_split # 移除 ConcatDataset
import numpy as np

# 假设这些模块在正确的位置
# 导入 build_vae_lmdb 
# 注意：你需要确保 utils 路径在 sys.path 中，例如在运行脚本前添加：
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
from utils.balanced_dataset_utils import build_vae_lmdb

from utils.dataset_utils import FixedDragMeshDatasetV2 


def main(args):
    print("\n" + "="*70)
    print("BUILDING LMDB DATABASES (IN-CATEGORY VALIDATION ONLY)")
    print("="*70)

    # 1. 加载配置文件
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 仅加载 train_categories，因为 val_cross_categories 应该为空或被忽略
    train_categories = config['train_categories'] 
    val_in_config = config.get('val_in_category', {})
    val_split_ratio = val_in_config.get('split_ratio', 0.1) # 假设 0.1
    seed = val_in_config.get('random_seed', 42)

    print(f"Total categories (Train + Val): {len(train_categories)}")
    print(f"Val split ratio (In-Category): {val_split_ratio} (Seed: {seed})")

    # 2. 加载所有类别的数据
    print("\nLoading all data...")
    full_dataset = FixedDragMeshDatasetV2(
        dataset_root=args.dataset_root,
        num_frames=args.num_frames,
        num_points=args.num_points,
        categories=train_categories
    )
    print(f"Found {len(full_dataset)} total samples.")

    # 3. 筛选数据（保持原有的筛选逻辑）
    print("Filtering data for strong intent (angle range > π/2)...")
    filtered_indices = []
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
        # 假设 qr_gt 的第一个元素是 w
        angles = 2 * torch.acos(torch.clamp(sample['qr_gt'][:, 0].abs(), 0, 1))

        if (angles.max() - angles.min()) > np.pi / 2:
            filtered_indices.append(i)
            
    if len(filtered_indices) < 100:
        print("Warning: Too few filtered, using all samples.")
        filtered_indices = list(range(len(full_dataset)))
        
    full_dataset = Subset(full_dataset, filtered_indices)
    print(f"Filtered to {len(full_dataset)} samples.")
    
    # 4. 执行训练集和验证集的拆分
    val_size = int(len(full_dataset) * val_split_ratio)
    train_size = len(full_dataset) - val_size

    print(f"\nSplitting into {train_size} (Train) and {val_size} (Val) samples.")
    
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 5. 构建训练 LMDB
    output_path_train = args.output_prefix + "_train.lmdb"
    print(f"\nBuilding TRAINING LMDB (Size: {len(train_subset)})...")
    build_vae_lmdb(
        dataset=train_subset, 
        output_path=output_path_train,
        categories=train_categories, 
        num_frames=args.num_frames,
        num_points=args.num_points
    )

    # 6. 构建验证 LMDB
    output_path_val = args.output_prefix + "_val.lmdb"
    print(f"\nBuilding VALIDATION LMDB (Size: {len(val_subset)})...")
    build_vae_lmdb(
        dataset=val_subset, 
        output_path=output_path_val,
        categories=train_categories, 
        num_frames=args.num_frames,
        num_points=args.num_points,
        # 移除了 val_in_size 参数
    )

    print("\n" + "="*70)
    print("LMDB BUILD COMPLETE (IN-DOMAIN FOCUS)!")
    print(f"Train Output: {output_path_train}")
    print(f"Val Output:   {output_path_val}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VAE LMDB (In-Category Split Only)")
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output LMDBs')
    parser.add_argument('--config', type=str,
                        default='config/category_split_v2_in_domain.json',
                        help='Category split configuration')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of trajectory frames')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points to sample')
    args = parser.parse_args()
    main(args)