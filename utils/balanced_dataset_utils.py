"""
File: utils/balanced_dataset_utils.py
Utilities for building/reading LMDB datasets and computing balanced sampling weights.
"""
import os
import torch
import numpy as np
import lmdb
import pickle
import gc
import random
import multiprocessing as mp
from typing import Optional, List, Dict, Tuple
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

def random_axis_rotation(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Random in-plane rotation augmentation."""
    angle = random.uniform(0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    
    keys_to_rotate = ['initial_mesh', 'drag_point', 'drag_vector', 'joint_axis', 
                      'joint_origin', 'rotation_direction', 'trajectory_vectors', 'drag_trajectory']
    for k in keys_to_rotate:
        if k in data:
            val = data[k]
            if isinstance(val, np.ndarray) and val.shape[-1] == 3:
                data[k] = val @ R.T
    return data

def get_motion_type_weights(dataset: Dataset, target_ratio: float = 3.0) -> torch.Tensor:
    """Compute class-balancing weights for WeightedRandomSampler."""
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    
    revolute_indices = []
    prismatic_indices = []

    print("Analyzing dataset motion types...")
    for idx_in_subset, i in enumerate(tqdm(indices)):
        try:
            sample = base_dataset[i]
            if sample is None: continue
            jt = sample['joint_type'].item() if hasattr(sample['joint_type'], 'item') else sample['joint_type']
            if jt == 0: revolute_indices.append(idx_in_subset)
            elif jt == 1: prismatic_indices.append(idx_in_subset)
        except: continue

    weights = torch.ones(len(dataset), dtype=torch.double)
    if not revolute_indices or not prismatic_indices:
        return weights

    n_rev, n_pri = len(revolute_indices), len(prismatic_indices)
    if n_rev / n_pri > target_ratio:
        w_rev, w_pri = (n_pri * target_ratio) / n_rev, 1.0
    else:
        w_rev, w_pri = 1.0, (n_rev / target_ratio) / n_pri
    
    weights[revolute_indices] = w_rev
    weights[prismatic_indices] = w_pri
    return weights

# --- Worker helpers (used by optional multiprocessing / sequential fallback) ---
def _worker_init(ds):
    global _global_dataset
    _global_dataset = ds

def _worker_process_sample(idx):
    """Worker-side sample serialization."""
    try:
        sample = _global_dataset[idx]
        if sample is None:
            return None
        
        serialized = {}
        keys_to_save = [
            'initial_mesh', 'drag_point', 'drag_vector', 'qr_gt', 'qd_gt', 
            'joint_type', 'joint_axis', 'joint_origin', 'part_mask',
            'rotation_direction', 'trajectory_vectors', 'drag_trajectory'
        ]
        for k in keys_to_save:
            if k in sample:
                val = sample[k]
                if isinstance(val, torch.Tensor):
                    serialized[k] = val.numpy()
                else:
                    serialized[k] = val
        return serialized
    except Exception:
        return None

def build_vae_lmdb(dataset, output_path, categories=None, num_frames=16, num_points=4096):
    """
    Build an LMDB for VAE training.

    Notes:
    - In some environments (especially after torch/OMP initialization), multiprocessing with the
      default fork start method may deadlock (symptoms: progress bar stuck; processes in futex_wait).
      Since the loader already supports low-memory streaming point sampling, the default here is a
      sequential build for stability.
    - Fixes the "'int' object has no attribute 'size'" error when parsing joint_type.
    """
    print("\n" + "="*70)
    print("BUILDING VAE LMDB DATABASE")
    print("="*70)
    
    total_samples = len(dataset)
    # Open LMDB environment (100GB map size by default).
    env = lmdb.open(output_path, map_size=1024**3 * 100) 
    
    actual_count = 0
    revolute_count = 0
    prismatic_count = 0

    # Default: sequential execution to avoid mp/fork deadlocks.
    global _global_dataset
    _global_dataset = dataset
    pbar = tqdm(range(total_samples), total=total_samples, desc="Writing LMDB")
    for idx in pbar:
        serialized = _worker_process_sample(idx)
        if serialized is None:
            continue

        # --- Robust joint_type parsing ---
        jt_val = serialized.get('joint_type')
        if jt_val is not None:
            # Handle numpy arrays
            if isinstance(jt_val, np.ndarray):
                jt = int(jt_val.item()) if jt_val.size == 1 else int(jt_val[0])
            # Handle python scalars / sequences (int, float, list, ...)
            else:
                jt = int(jt_val[0]) if isinstance(jt_val, (list, tuple)) else int(jt_val)
            
            if jt == 0: revolute_count += 1
            else: prismatic_count += 1

        # Write LMDB entry.
        with env.begin(write=True) as txn:
            key = f"{actual_count:08d}".encode('ascii')
            txn.put(key, pickle.dumps(serialized))
            actual_count += 1
        
        del serialized
        if actual_count % 100 == 0:
            gc.collect()

    # Write metadata.
    with env.begin(write=True) as txn:
        metadata = {
            'num_samples': actual_count, 'num_frames': num_frames, 'num_points': num_points,
            'categories': categories, 'revolute_count': revolute_count, 'prismatic_count': prismatic_count
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))

    env.close()
    print(f"\nDone! Saved {actual_count} samples.")

class VAE_LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, augment: bool = False):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.augment = augment
        with self.env.begin(write=False) as txn:
            meta = txn.get(b'__metadata__')
            if meta is None: raise ValueError("Corrupt LMDB!")
            self.metadata = pickle.loads(meta)
        self.num_samples = self.metadata['num_samples']

    def __len__(self): return self.num_samples
            
    def __getitem__(self, idx: int):
        key = f"{idx:08d}".encode('ascii')
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
        if not data: return None
        
        np_data = pickle.loads(data)
        
        if self.augment and random.random() < 0.5:
            np_data = random_axis_rotation(np_data)
        
        res = {}
        for k, v in np_data.items():
            if isinstance(v, np.ndarray):
                if v.dtype == np.float64: v = v.astype(np.float32)
                res[k] = torch.from_numpy(v)
            else:
                res[k] = torch.tensor(v)
        
        if 'joint_type' in res: res['joint_type'] = res['joint_type'].long()
        res['is_cross_category'] = torch.tensor(0).long()
        return res