# -------------------------------------------------------------------
# scripts/build_lmdb.py
# -------------------------------------------------------------------
"""
Build LMDB databases for VAE/KPP training.

This script samples point clouds in a low-memory mode, performs normalization, and writes
train/val splits to LMDB for efficient training I/O.
"""
import sys
import os
import argparse
import json
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset, random_split, Subset

# Ensure modules and utils are in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the GAPartNet loader.
from modules.data_loader_v2 import GAPartNetLoaderV2
from utils.balanced_dataset_utils import build_vae_lmdb

class SmartProcessingDataset(Dataset):
    """
    Smart processing dataset:
    1) Category-based filtering (config-driven)
    2) Uses the GAPartNet loader
    3) Filtering based on motion magnitude
    4) Normalization and point sampling
    """
    def __init__(self, root, categories=None, num_frames=16, num_points=4096, joint_selection: str = "largest_motion"):
        print(f"Initializing Loader from: {root}")
        self.loader = GAPartNetLoaderV2(root)
        self.num_frames = num_frames
        self.num_points = num_points
        self.joint_selection = joint_selection
        
        # === 1) Category-based filtering ===
        if categories:
            print(f"Filtering for {len(categories)} categories...")
            self.valid_indices = []
            for i, obj_path in enumerate(self.loader.object_list):
                # Check meta.json for category labels.
                meta_path = os.path.join(obj_path, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            if meta.get('model_cat') in categories:
                                self.valid_indices.append(i)
                    except:
                        pass
            print(f"Found {len(self.valid_indices)} objects matching categories.")
        else:
            self.valid_indices = list(range(len(self.loader.object_list)))
            
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map back to the loader's global object index.
        real_idx = self.valid_indices[idx]
        
        try:
            # 1) Fetch a raw sample from the loader.
            raw = self.loader.generate_training_sample(
                real_idx,
                self.num_frames,
                joint_selection=self.joint_selection,
                return_mesh=False,          # key: low-memory point-cloud mode (avoid mesh concat OOM)
                num_points=self.num_points  # sample points inside the loader
            )
            
            # 2) Motion-based filtering.
            joint_type = raw['joint_type']
            is_valid = False
            
            if joint_type in ['revolute', 'continuous']:
                # Check rotation magnitude via quaternion angle difference (first vs last frame).
                q_s, q_e = raw['qr_sequence'][0], raw['qr_sequence'][-1]
                dot = np.abs(np.dot(q_s, q_e))
                angle_diff = 2 * np.arccos(np.clip(dot, -1, 1))
                # Threshold: > ~17 degrees (~0.3 rad).
                if angle_diff > 0.3: 
                    is_valid = True
            
            elif joint_type == 'prismatic':
                # Check translation distance.
                traj = raw['drag_trajectory']
                dist = np.linalg.norm(traj[-1] - traj[0])
                if dist > 0.1: 
                    is_valid = True
            
            if not is_valid: 
                return None  # tell build_vae_lmdb to skip this sample
            
            # 3) === Normalization (points already sampled by the loader) ===
            points = raw['initial_mesh']  # [N,3] ndarray
            center = raw['joint_origin']  # center at the joint origin

            # Compute scale: prefer loader-provided bounds; otherwise use point-cloud bounds.
            if 'bounds_min' in raw and 'bounds_max' in raw:
                bmin = raw['bounds_min']
                bmax = raw['bounds_max']
            else:
                bmin = points.min(axis=0)
                bmax = points.max(axis=0)
            scale = (bmax - bmin).max()
            if scale < 1e-6: scale = 1.0

            points = ((points - center) / scale).astype(np.float32)

            # 4) Part mask is already point-level (0=static, 1=movable).
            p_mask = raw['part_mask'].astype(np.int32)

            # 5) Pack output dict
            out = {
                'initial_mesh': points.astype(np.float32, copy=False),
                'drag_point': ((raw['drag_point'] - center) / scale).astype(np.float32),
                'drag_vector': (raw['drag_vector'] / scale).astype(np.float32),
                'qr_gt': raw['qr_sequence'].astype(np.float32),
                'qd_gt': (raw['qd_sequence'] / scale).astype(np.float32),
                'joint_type': 0 if joint_type in ['revolute', 'continuous'] else 1,
                'joint_axis': raw['joint_axis'].astype(np.float32),
                'joint_origin': ((raw['joint_origin'] - center) / scale).astype(np.float32), 
                'part_mask': p_mask.astype(np.float32),
                
                # Extra fields (used by loss/augmentation)
                'rotation_direction': raw['rotation_direction'].astype(np.float32),
                'trajectory_vectors': (raw['trajectory_vectors'] / scale).astype(np.float32),
                'drag_trajectory': ((raw['drag_trajectory'] - center) / scale).astype(np.float32)
            }
            return out

        except Exception as e:
            # print(f"Skip sample {real_idx}: {e}")
            return None

def main(args):
    print("\n" + "="*70)
    print("BUILDING LMDB DATABASES (IN-CATEGORY VALIDATION ONLY)")
    print("With motion-based filtering and low-memory point sampling")
    print("="*70)

    # 1) Load configuration.
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load train_categories only.
    train_categories = config['train_categories'] 
    val_in_config = config.get('val_in_category', {})
    val_split_ratio = val_in_config.get('split_ratio', 0.1)
    seed = val_in_config.get('random_seed', 42)

    print(f"Total categories: {len(train_categories)}")
    print(f"Val split ratio: {val_split_ratio}")

    # 2) Initialize smart dataset (includes filtering in __getitem__).
    # Note: we do not use FixedDragMeshDatasetV2 here; we use SmartProcessingDataset to avoid
    # unnecessary work performed during dataset initialization.
    print("\nInitializing Smart Dataset...")
    full_dataset = SmartProcessingDataset(
        root=args.dataset_root,
        categories=train_categories,
        num_frames=args.num_frames,
        num_points=args.num_points,
        joint_selection=args.joint_selection
    )
    
    # 3) Note: len(full_dataset) counts objects matching categories; the number of valid samples is
    # determined during LMDB build because __getitem__ may return None.
    print(f"Found {len(full_dataset)} matching objects (filtering happens during build).")

    # 4) Split into train/val objects.
    val_size = int(len(full_dataset) * val_split_ratio)
    train_size = len(full_dataset) - val_size

    print(f"\nSplitting into {train_size} (Train) and {val_size} (Val) objects.")
    
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # 5) Build training LMDB.
    output_path_train = args.output_prefix + "_train.lmdb"
    print(f"\nBuilding TRAINING LMDB...")
    build_vae_lmdb(
        dataset=train_subset, 
        output_path=output_path_train,
        categories=train_categories, 
        num_frames=args.num_frames,
        num_points=args.num_points
    )

    # 6) Build validation LMDB.
    output_path_val = args.output_prefix + "_val.lmdb"
    print(f"\nBuilding VALIDATION LMDB...")
    build_vae_lmdb(
        dataset=val_subset, 
        output_path=output_path_val,
        categories=train_categories, 
        num_frames=args.num_frames,
        num_points=args.num_points
    )

    print("\n" + "="*70)
    print("LMDB BUILD COMPLETE!")
    print(f"Train Output: {output_path_train}")
    print(f"Val Output:   {output_path_val}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build VAE LMDB")
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output LMDBs')
    parser.add_argument('--config', type=str,
                        default='config/category_split_v2.json',
                        help='Category split configuration')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of trajectory frames')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points to sample')
    parser.add_argument(
        '--joint_selection',
        type=str,
        default='largest_motion',
        choices=['largest_motion', 'random', 'first'],
        help='How to choose the joint for each object when generating samples'
    )
    args = parser.parse_args()
    main(args)