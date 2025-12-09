import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F


def axis_angle_to_dualquat(axis: np.ndarray, origin: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert axis-angle representation to dual quaternion for revolute joint.

    Args:
        axis: Rotation axis [3]
        origin: Point on rotation axis [3]
        angle: Rotation angle in radians

    Returns:
        qr: Rotation quaternion [4]
        qd: Translation quaternion [4]
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # Rotation quaternion from axis-angle
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    qr = np.array([cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])

    # Translation quaternion
    # qd = 0.5 * quaternion_mul(t, qr)
    # For revolute joint, translation is from rotation around point
    t_vec = origin - quaternion_rotate(qr, origin)
    t = np.array([0.0, t_vec[0], t_vec[1], t_vec[2]])

    qd = 0.5 * quaternion_multiply(t, qr)

    return qr, qd


def translation_to_dualquat(axis: np.ndarray, distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert translation to dual quaternion for prismatic joint.

    Args:
        axis: Translation direction [3]
        distance: Translation distance

    Returns:
        qr: Rotation quaternion [4] (identity)
        qd: Translation quaternion [4]
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # No rotation for prismatic joint
    qr = np.array([1.0, 0.0, 0.0, 0.0])

    # Translation vector
    t_vec = axis * distance
    t = np.array([0.0, t_vec[0], t_vec[1], t_vec[2]])

    # qd = 0.5 * quaternion_mul(t, qr) = 0.5 * t (since qr is identity)
    qd = 0.5 * t

    return qr, qd


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate(q: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply quaternion rotation to a 3D point."""
    p = np.array([0.0, point[0], point[1], point[2]])
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, p), q_conj)
    return rotated[1:]


class GAPartNetLoaderV2:
    """
    Data loader for GAPartNet with Dual Quaternion ground truth.
    """

    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.object_list = self._get_object_list()

    def _get_object_list(self) -> List[str]:
        """Get list of all object directories in the dataset."""
        object_list = []
        for category in os.listdir(self.dataset_root):
            category_path = os.path.join(self.dataset_root, category)
            if not os.path.isdir(category_path):
                continue
            for obj_id in os.listdir(category_path):
                obj_path = os.path.join(category_path, obj_id)
                urdf_path = os.path.join(obj_path, "mobility_annotation_gapartnet.urdf")
                if os.path.exists(urdf_path):
                    object_list.append(obj_path)
        return object_list

    def parse_urdf(self, urdf_path: str) -> Dict:
        """Parse URDF file to extract joint information."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        joints = []
        links = {}

        # Parse links and extract ALL mesh filenames
        for link in root.findall('link'):
            link_name = link.get('name')
            if link_name == 'base' or link_name == 'root':
                continue

            # Find ALL visual elements (a link can have multiple meshes!)
            mesh_files = []
            for visual in link.findall('visual'):
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        if mesh_filename not in mesh_files:  # Avoid duplicates
                            mesh_files.append(mesh_filename)

            if len(mesh_files) > 0:
                # Store all mesh files for this link
                links[link_name] = mesh_files

        # Parse joints
        for joint in root.findall('joint'):
            joint_type = joint.get('type')
            if joint_type == 'fixed':
                continue

            joint_name = joint.get('name')
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            axis_elem = joint.find('axis')
            if axis_elem is not None:
                axis = [float(x) for x in axis_elem.get('xyz').split()]
            else:
                axis = [0, 0, 1]

            origin_elem = joint.find('origin')
            if origin_elem is not None:
                xyz_str = origin_elem.get('xyz')
                rpy_str = origin_elem.get('rpy')
                xyz = [float(x) for x in xyz_str.split()] if xyz_str else [0, 0, 0]
                rpy = [float(x) for x in rpy_str.split()] if rpy_str else [0, 0, 0]
            else:
                xyz = [0, 0, 0]
                rpy = [0, 0, 0]

            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower = float(limit_elem.get('lower'))
                upper = float(limit_elem.get('upper'))
            else:
                lower = -1.57
                upper = 1.57

            joints.append({
                'name': joint_name,
                'type': joint_type,
                'parent': parent,
                'child': child,
                'axis': np.array(axis),
                'origin_xyz': np.array(xyz),
                'origin_rpy': np.array(rpy),
                'limit': {'lower': lower, 'upper': upper}
            })

        return {'joints': joints, 'links': links}

    def load_part_meshes(self, obj_path: str, links: Dict[str, List[str]]) -> Dict[str, trimesh.Trimesh]:
        """
        Load meshes for each part/link.

        Args:
            obj_path: Path to object directory
            links: Dict mapping link_name to list of mesh filenames

        Returns:
            Dict mapping link_name to combined trimesh (all meshes for that link merged)
        """
        part_meshes = {}

        for link_name, mesh_filenames in links.items():
            # Load all meshes for this link
            link_meshes = []
            for mesh_filename in mesh_filenames:
                mesh_path = os.path.join(obj_path, mesh_filename)
                if os.path.exists(mesh_path):
                    try:
                        mesh = trimesh.load(mesh_path, force='mesh', process=False)
                        link_meshes.append(mesh)
                    except Exception as e:
                        print(f"Warning: Failed to load {mesh_path}: {e}")

            # Combine all meshes for this link
            if len(link_meshes) > 0:
                if len(link_meshes) == 1:
                    part_meshes[link_name] = link_meshes[0]
                else:
                    # Concatenate multiple meshes for this link
                    combined_mesh = trimesh.util.concatenate(link_meshes)
                    part_meshes[link_name] = combined_mesh

        return part_meshes

    def generate_training_sample(self, obj_idx: int, num_frames: int = 16) -> Dict:
        """
        Generate training sample with dual quaternion ground truth.

        Returns:
            Dictionary containing:
                - initial_mesh: Combined mesh at initial state
                - drag_point: Starting point of drag [3]
                - drag_vector: Drag displacement vector [3]
                - qr_sequence: Rotation quaternion sequence [num_frames, 4]
                - qd_sequence: Translation quaternion sequence [num_frames, 4]
                - joint_type: 'revolute' or 'prismatic'
                - part_mask: Which vertices belong to movable part [N]
        """
        obj_path = self.object_list[obj_idx]

        # Parse URDF
        urdf_path = os.path.join(obj_path, "mobility_annotation_gapartnet.urdf")
        urdf_data = self.parse_urdf(urdf_path)

        # Load part meshes
        part_meshes = self.load_part_meshes(obj_path, urdf_data['links'])

        if len(urdf_data['joints']) == 0:
            raise ValueError(f"No movable joints found in {obj_path}")

        # Select first movable joint
        joint = urdf_data['joints'][0]
        child_link = joint['child']

        if child_link not in part_meshes:
            raise ValueError(f"Child link {child_link} mesh not found")

        child_mesh = part_meshes[child_link]

        if joint['type'] in ['revolute', 'continuous']:
            distances = np.linalg.norm(child_mesh.vertices - joint['origin_xyz'], axis=1)
            far_indices = np.argsort(distances)[-max(1, len(child_mesh.vertices) // 5):]  # Top 20%
            drag_point_idx = np.random.choice(far_indices)
            drag_point_start = child_mesh.vertices[drag_point_idx].copy()

            # Improve angle range
            lower, upper = joint['limit']['lower'], joint['limit']['upper']
            if upper - lower < np.pi:
                lower, upper = -np.pi, np.pi
            angles = np.linspace(lower, upper, num_frames)
            
            # For rotation_direction
            angle_change = upper - lower  # Always positive if upper > lower
            rotation_direction = np.array(joint['axis']) * np.sign(angle_change)  # Ensure positive

        else:
            drag_point_idx = np.random.randint(0, len(child_mesh.vertices))
            drag_point_start = child_mesh.vertices[drag_point_idx].copy()

        # Generate dual quaternion sequence
        lower, upper = joint['limit']['lower'], joint['limit']['upper']

        if joint['type'] == 'continuous':
            upper = max(upper, np.pi)  
            lower = min(lower, -np.pi)

        if upper > lower:
            angles = np.linspace(lower, upper, num_frames)
        else:
            if joint['type'] == 'continuous':
                angles = np.linspace(-np.pi, np.pi, num_frames)
            else:
                angles = np.linspace(0, np.pi/2, num_frames)  

        print(f"Joint type: {joint['type']}, Range: [{lower:.3f}, {upper:.3f}], Generated angles: {len(angles)} frames")

        qr_sequence = []
        qd_sequence = []

        for angle in angles:
            if joint['type'] == 'revolute' or joint['type'] == 'continuous':
                # continuous joint is like revolute but without limits
                qr, qd = axis_angle_to_dualquat(
                    joint['axis'],
                    joint['origin_xyz'],
                    angle
                )
            elif joint['type'] == 'prismatic':
                qr, qd = translation_to_dualquat(joint['axis'], angle)
            else:
                raise ValueError(f"Unknown joint type: {joint['type']}")

            qr_sequence.append(qr)
            qd_sequence.append(qd)

        qr_sequence = np.array(qr_sequence)  # [num_frames, 4]
        qd_sequence = np.array(qd_sequence)  # [num_frames, 4]

        # Combine all parts into initial mesh
        static_meshes = [mesh for name, mesh in part_meshes.items() if name != child_link]
        all_meshes = static_meshes + [child_mesh]
        initial_mesh = trimesh.util.concatenate(all_meshes)

        # Create part mask (which vertices are movable)
        num_static_verts = sum(mesh.vertices.shape[0] for mesh in static_meshes)
        num_movable_verts = child_mesh.vertices.shape[0]
        part_mask = np.concatenate([
            np.zeros(num_static_verts, dtype=np.int32),
            np.ones(num_movable_verts, dtype=np.int32)
        ])

        # Calculate drag vector with trajectory information
        # Apply dual quaternion sequence to drag point to get trajectory
        drag_trajectory = []
        for qr, qd in zip(qr_sequence, qd_sequence):
            # Apply current transformation to drag point
            point_rotated = quaternion_rotate(qr, drag_point_start)
            t_vec = 2.0 * quaternion_multiply(qd, quaternion_conjugate(qr))[1:]
            point_transformed = point_rotated + t_vec
            drag_trajectory.append(point_transformed)

        drag_trajectory = np.array(drag_trajectory)  # [num_frames, 3]

        # Multi-tangent: start, mid, end diffs
        if len(drag_trajectory) > 2:
            start_tangent = drag_trajectory[1] - drag_trajectory[0]
            mid_idx = len(drag_trajectory) // 2
            mid_tangent = drag_trajectory[mid_idx+1] - drag_trajectory[mid_idx]
            end_tangent = drag_trajectory[-1] - drag_trajectory[-2]
            trajectory_vectors = np.stack([start_tangent, mid_tangent, end_tangent])  # [3,3]
        else:
            trajectory_vectors = drag_trajectory[1:] - drag_trajectory[:-1]

        # Improved drag_vector: average tangents * total_length
        diffs = drag_trajectory[1:] - drag_trajectory[:-1]
        lengths = np.linalg.norm(diffs, axis=1)
        total_length = lengths.sum()
        if total_length > 0:
            avg_dir = np.mean(trajectory_vectors, axis=0)  # Average multi-tangents
            avg_dir = avg_dir / np.linalg.norm(avg_dir)
            drag_vector = avg_dir * total_length
        else:
            drag_vector = drag_trajectory[-1] - drag_trajectory[0]

        rotation_direction = None
        if joint['type'] in ['revolute', 'continuous']:
            if len(angles) >= 2:
                angle_change = angles[-1] - angles[0]
                rotation_direction = np.array(joint['axis']) * np.sign(angle_change)
            else:
                rotation_direction = np.array(joint['axis']) 

        drag_dir = drag_vector / (np.linalg.norm(drag_vector) + 1e-8)
        relative_point = drag_point_start - joint['origin_xyz']
        cross_axis_drag = np.cross(joint['axis'], drag_dir)
        rotation_sign = np.sign(np.dot(cross_axis_drag, relative_point)) or 1.0  # Positive if dot >0, else negative

        rotation_direction = np.array(joint['axis']) * rotation_sign

        # Apply sign to drag_vector
        drag_vector = drag_vector * rotation_sign  # If negative, reverse vector

        return {
            'initial_mesh': initial_mesh,
            'drag_point': drag_point_start,
            'drag_vector': drag_vector,
            'trajectory_vectors': trajectory_vectors,  
            'rotation_direction': rotation_direction,  
            'qr_sequence': qr_sequence,
            'qd_sequence': qd_sequence,
            'joint_type': joint['type'],
            'part_mask': part_mask,
            'obj_path': obj_path,
            'joint_axis': joint['axis'],  
            'joint_origin': joint['origin_xyz']  
        }

    def __len__(self):
        return len(self.object_list)


class DragMeshDatasetV2(torch.utils.data.Dataset):
    """
    PyTorch Dataset with Dual Quaternion ground truth.
    """

    def __init__(self, dataset_root: str, num_frames: int = 16, num_points: int = 4096):
        self.loader = GAPartNetLoaderV2(dataset_root)
        self.num_frames = num_frames
        self.num_points = num_points

    def mesh_to_pointcloud(self, mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
        """Sample point cloud from mesh surface."""
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        return points

    def normalize_mesh(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
        """
        Normalize mesh to unit bounding box centered at origin.
        Returns normalized mesh, center, and scale for denormalization.
        """
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        scale = (bounds[1] - bounds[0]).max()
        
        if scale < 1e-6:
            scale = 1.0

        mesh_normalized = mesh.copy()
        mesh_normalized.vertices = (mesh_normalized.vertices - center) / scale
        return mesh_normalized, center, scale

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        """
        sample = self.loader.generate_training_sample(idx, self.num_frames)

        center = sample['joint_origin'] 
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

        rotation_direction = sample.get('rotation_direction')
        if rotation_direction is not None:
            rotation_direction = np.array(rotation_direction)  

        trajectory_vectors = sample.get('trajectory_vectors')
        if trajectory_vectors is not None:
            trajectory_vectors = np.array(trajectory_vectors)  
        
        joint_origin_normalized = (sample['joint_origin'] - center) / scale

        if sample['joint_type'] == 'revolute' or sample['joint_type'] == 'continuous':
            joint_type = 0
        elif sample['joint_type'] == 'prismatic':
            joint_type = 1
        else:
            joint_type = 0  # default to revolute
        
        qr_gt = sample['qr_sequence']
        
        qd_gt = sample['qd_sequence'].copy()
        qd_gt = qd_gt / scale
        
        joint_axis = sample['joint_axis']
        joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-8)
        

        original_mesh = sample['initial_mesh']
        vertex_part_mask = sample['part_mask'] # [N_verts]
        
        sampled_part_mask = np.zeros(self.num_points, dtype=np.int32)
        for i, fid in enumerate(face_idx):
            face_vertices = original_mesh.faces[fid]
            face_mask_values = vertex_part_mask[face_vertices]
            sampled_part_mask[i] = np.bincount(face_mask_values.astype(int)).argmax()
    

        result = {
            'initial_mesh': torch.from_numpy(initial_pc).float(),
            'drag_point': torch.from_numpy(drag_point).float(),
            'drag_vector': torch.from_numpy(drag_vector).float(),
            'qr_gt': torch.from_numpy(qr_gt).float(),
            'qd_gt': torch.from_numpy(qd_gt).float(),
            'joint_type': torch.tensor(joint_type).long(),
            'joint_axis': torch.from_numpy(joint_axis).float(),
            'joint_origin': torch.from_numpy(joint_origin_normalized).float(), 
            'part_mask': torch.from_numpy(sampled_part_mask).float()
        }

        if rotation_direction is not None:
            result['rotation_direction'] = torch.from_numpy(rotation_direction).float()

        if trajectory_vectors is not None:
            result['trajectory_vectors'] = torch.from_numpy(trajectory_vectors).float()

        return result