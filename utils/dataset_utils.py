"""
Shared dataset utilities for category-based data loading.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import torch
import numpy as np
from typing import Optional, List, Dict

from modules.data_loader_v2 import GAPartNetLoaderV2, DragMeshDatasetV2


class FixedGAPartNetLoader(GAPartNetLoaderV2):
    """Fixed loader for flat directory structure with category filtering."""

    def __init__(self, dataset_root: str, categories: Optional[List[str]] = None):
        self.categories = categories
        super().__init__(dataset_root)

    def _get_object_list(self):
        """Get list of all object directories in the dataset (flat structure)."""
        object_list = []
        for obj_id in os.listdir(self.dataset_root):
            obj_path = os.path.join(self.dataset_root, obj_id)
            if not os.path.isdir(obj_path):
                continue
            urdf_path = os.path.join(obj_path, "mobility_annotation_gapartnet.urdf")
            if not os.path.exists(urdf_path):
                continue

            # Check category filtering
            if self.categories is not None:
                meta_path = os.path.join(obj_path, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            category = meta.get('model_cat', None)
                            if category not in self.categories:
                                continue
                    except:
                        continue
                else:
                    continue

            object_list.append(obj_path)
        return object_list


class FixedDragMeshDatasetV2(DragMeshDatasetV2):
    """Dataset with fixed loader and category filtering."""

    def __init__(self, dataset_root: str, num_frames: int = 16, num_points: int = 4096,
                 categories: Optional[List[str]] = None):
        self.loader = FixedGAPartNetLoader(dataset_root, categories=categories)
        self.num_frames = num_frames
        self.num_points = num_points


class KPPDataset(FixedDragMeshDatasetV2):
    """Dataset for KPP-Net training with part_mask and joint parameters."""

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        """
        sample = self.loader.generate_training_sample(idx, self.num_frames)

        # 1. 我们的 "中心" 现在是关节原点
        center = sample['joint_origin'] 
        
        # 2. 尺度仍然使用包围盒
        bounds = sample['initial_mesh'].bounds
        scale = (bounds[1] - bounds[0]).max()
        if scale < 1e-6:
            scale = 1.0

        mesh_normalized_verts = (sample['initial_mesh'].vertices - center) / scale
        mesh_normalized = trimesh.Trimesh(vertices=mesh_normalized_verts, 
                                          faces=sample['initial_mesh'].faces)
        
        initial_pc, face_idx = trimesh.sample.sample_surface(mesh_normalized, self.num_points)

        drag_point = (sample['drag_point'] - center) / scale 
        drag_vector = sample['drag_vector'] / scale
        
        joint_origin_normalized = (sample['joint_origin'] - center) / scale

        
        #  Joint type
        if sample['joint_type'] == 'revolute' or sample['joint_type'] == 'continuous':
            joint_type = 0
        elif sample['joint_type'] == 'prismatic':
            joint_type = 1
        else:
            joint_type = 0  # default to revolute
        
        qr_gt = sample['qr_sequence']

        if joint_type == 0:
            qd_gt = sample['qd_sequence'].copy() 
            qd_gt = qd_gt / scale
        else:
            qd_gt = sample['qd_sequence'].copy()
            qd_gt = qd_gt / scale

        # 8. Joint axis (不变, 归一化)
        joint_axis = sample['joint_axis']
        joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-8)
        
        # 9. Part mask (不变)
        original_mesh = sample['initial_mesh']
        vertex_part_mask = sample['part_mask'] # [N_verts]
        
        sampled_part_mask = np.zeros(self.num_points, dtype=np.int32)
        for i, fid in enumerate(face_idx):
            face_vertices = original_mesh.faces[fid]
            face_mask_values = vertex_part_mask[face_vertices]
            sampled_part_mask[i] = np.bincount(face_mask_values.astype(int)).argmax()
    

        return {
            'initial_mesh': torch.from_numpy(initial_pc).float(),
            'drag_point': torch.from_numpy(drag_point).float(),
            'drag_vector': torch.from_numpy(drag_vector).float(),
            'qr_gt': torch.from_numpy(qr_gt).float(),
            'qd_gt': torch.from_numpy(qd_gt).float(),
            'joint_type': torch.tensor(joint_type).long(),
            'joint_axis': torch.from_numpy(joint_axis).float(),
            'joint_origin': torch.from_numpy(joint_origin_normalized).float(), # 发送 (0,0,0)
            'part_mask': torch.from_numpy(sampled_part_mask).float()
        }

    def mesh_to_pointcloud_with_faces(self, mesh, num_points):
        """Sample point cloud from mesh surface and return face indices."""
        import trimesh
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        return points, face_idx

    def map_face_mask_to_points(self, mesh, part_mask_vertices, face_indices):
        """Map vertex-based part mask to sampled points using face indices."""
        # For each sampled point (via face), get the mask value
        part_mask_points = np.zeros(len(face_indices), dtype=np.float32)

        for i, face_idx in enumerate(face_indices):
            # Get vertices of this face
            face_vertices = mesh.faces[face_idx]
            # Use majority vote or first vertex mask value
            face_mask_values = part_mask_vertices[face_vertices]
            part_mask_points[i] = np.mean(face_mask_values)  # Average mask value for this face

        return part_mask_points


def load_category_config(config_path: str):
    """
    Load category split configuration from JSON file.

    Args:
        config_path: Path to category_split.json

    Returns:
        tuple of (train_categories, val_categories)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Category config not found: {config_path}")

    with open(config_path, 'r') as f:
        category_config = json.load(f)

    train_categories = category_config['train_categories']
    val_categories = category_config['val_categories']

    return train_categories, val_categories


def print_category_split_info(config_path: str, train_categories: List[str], val_categories: List[str]):
    """Print information about category split."""
    print("\n" + "="*60)
    print("CATEGORY-BASED DATA SPLIT")
    print("="*60)
    print(f"Config loaded from: {config_path}")
    print(f"Training categories ({len(train_categories)}): {train_categories}")
    print(f"Validation categories ({len(val_categories)}): {val_categories}")
    print("="*60 + "\n")
