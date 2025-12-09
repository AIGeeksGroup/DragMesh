"""
Build LMDB database for fast VAE training (FIXED for Hybrid Validation).

This script reads the v2 config file and correctly splits the data into:
1.  [prefix]_train.lmdb: Contains only the training split (e.g., 80%)
    of the training categories.
2.  [prefix]_val.lmdb: Contains the validation split (e.g., 20%)
    of the training categories (In-Category) AND all samples
    from the cross-validation categories (Cross-Category).

Usage:
    python scripts/build_lmdb.py \
        --dataset_root /path/to/partnet_mobility_part \
        --output_prefix /path/to/vae_data \
        --config config/category_split_v2.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import torch
from torch.utils.data import Subset, ConcatDataset, random_split

# ✅ 关键导入:
# 导入 build_vae_lmdb (必须是已修复的版本)
from utils.balanced_dataset_utils import build_vae_lmdb
# 导入 FixedDragMeshDatasetV2 (用于加载数据)
from utils.dataset_utils import FixedDragMeshDatasetV2 


def main(args):
    print("\n" + "="*70)
    print("BUILDING LMDB DATABASES (HYBRID VALIDATION V2)")
    print("="*70)

    # 1. 加载配置文件
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    train_categories = config['train_categories']
    val_cross_categories = config['val_cross_category']['categories']
    val_in_config = config.get('val_in_category', {})
    val_split_ratio = val_in_config.get('split_ratio', 0.2)
    seed = val_in_config.get('random_seed', 42)

    print(f"Train categories: {len(train_categories)}")
    print(f"Cross-Val categories: {len(val_cross_categories)}")
    print(f"In-Val split ratio: {val_split_ratio} (Seed: {seed})")

    # 2. 加载“训练”类别的所有数据
    print("\nLoading all 'train_categories' data...")
    full_train_dataset = FixedDragMeshDatasetV2(
        dataset_root=args.dataset_root,
        num_frames=args.num_frames,
        num_points=args.num_points,
        categories=train_categories
    )
    print(f"Found {len(full_train_dataset)} total samples in 'train_categories'.")

    # 3. 执行 In-Category 拆分
    val_in_size = int(len(full_train_dataset) * val_split_ratio)
    train_size = len(full_train_dataset) - val_in_size

    print(f"Splitting into {train_size} (Train) and {val_in_size} (Val-In) samples.")
    
    # 拆分为两个子集 (Subset)
    train_subset, val_in_subset = random_split(
        full_train_dataset,
        [train_size, val_in_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 4. 加载“交叉验证”类别的数据
    print("\nLoading 'val_cross_category' data...")
    val_cross_dataset = FixedDragMeshDatasetV2(
        dataset_root=args.dataset_root,
        num_frames=args.num_frames,
        num_points=args.num_points,
        categories=val_cross_categories
    )
    print(f"Found {len(val_cross_dataset)} total samples in 'val_cross_category'.")

    # 5. (关键) 合并验证集
    full_val_dataset = ConcatDataset([val_in_subset, val_cross_dataset])
    print(f"Total validation samples (In + Cross): {len(full_val_dataset)}")
    
    # 为元数据准备类别列表
    # (注意: ConcatDataset 没有 .categories, 我们手动合并)
    val_categories_list = train_categories + val_cross_categories

    # 6. 构建训练 LMDB
    output_path_train = args.output_prefix + "_train.lmdb"
    print(f"\nBuilding TRAINING LMDB (Size: {len(train_subset)})...")
    build_vae_lmdb(
        dataset=train_subset, # ✅ 传入 Subset
        output_path=output_path_train,
        categories=train_categories, # 仅用于元数据
        num_frames=args.num_frames,
        num_points=args.num_points
        # (训练集不需要 val_in_size)
    )

    # 7. 构建验证 LMDB
    output_path_val = args.output_prefix + "_val.lmdb"
    print(f"\nBuilding VALIDATION LMDB (Size: {len(full_val_dataset)})...")
    build_vae_lmdb(
        dataset=full_val_dataset, # ✅ 传入 ConcatDataset
        output_path=output_path_val,
        categories=val_categories_list, # 仅用于元数据
        num_frames=args.num_frames,
        num_points=args.num_points,
        val_in_size=val_in_size  # ✅ <--- ！！！这行是修复 Bug 的关键！！！
    )

    print("\n" + "="*70)
    print("LMDB BUILD COMPLETE!")
    print(f"Train Output: {output_path_train}")
    print(f"Val Output:   {output_path_val}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VAE LMDB (Hybrid Split V2)")

    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory of dataset')
    
    # ✅ 添加了 --output_prefix
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output LMDBs (e.g., "dataset_vae" -> "dataset_vae_train.lmdb")')
    
    parser.add_argument('--config', type=str,
                        default='config/category_split_v2.json',
                        help='Category split configuration (v2 required)')
    
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of trajectory frames')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points to sample')

    args = parser.parse_args()
    main(args)