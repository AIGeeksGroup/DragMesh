import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from collections import deque
import gc
import torch
import torch.nn.functional as F



def axis_angle_to_dualquat(axis: np.ndarray, origin: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert axis-angle representation to dual quaternion for revolute joint.
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    # Rotation quaternion from axis-angle
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    qr = np.array([cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])

    # Translation quaternion
    t_vec = origin - quaternion_rotate(qr, origin)
    t = np.array([0.0, t_vec[0], t_vec[1], t_vec[2]])

    qd = 0.5 * quaternion_multiply(t, qr)

    return qr, qd


def translation_to_dualquat(axis: np.ndarray, distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert translation to dual quaternion for prismatic joint.
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
        # Support the case where dataset_root directly points to an object folder.
        if os.path.exists(os.path.join(self.dataset_root, "mobility_annotation_gapartnet.urdf")):
            return [self.dataset_root]

        for category in os.listdir(self.dataset_root):
            category_path = os.path.join(self.dataset_root, category)
            if not os.path.isdir(category_path):
                continue
            
            # Support the case where the category directory itself contains the URDF.
            if os.path.exists(os.path.join(category_path, "mobility_annotation_gapartnet.urdf")):
                object_list.append(category_path)
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
            # Keep the link entry even if it has no mesh: it can still be a joint attachment.
            mesh_files = []
            for visual in link.findall('visual'):
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_filename = mesh.get('filename')
                        if mesh_filename not in mesh_files:  # Avoid duplicates
                            mesh_files.append(mesh_filename)
            links[link_name] = mesh_files

        # Parse joints (only movable joints are added to the `joints` list).
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

        # --- Build a full tree map (including fixed joints) ---
        # This is important for propagating motion to descendant links (e.g., handle follows door).
        full_tree_map = {}
        for joint in root.findall('joint'):
            p = joint.find('parent').get('link')
            c = joint.find('child').get('link')
            if p not in full_tree_map:
                full_tree_map[p] = []
            full_tree_map[p].append(c)

        return {'joints': joints, 'links': links, 'full_tree_map': full_tree_map}

    def _get_subtree_links(self, root_link: str, tree_map: Dict[str, List[str]]) -> List[str]:
        """
        Collect all descendant links starting from `root_link` (including itself).
        """
        subtree: List[str] = []
        seen = set()
        queue = deque([root_link])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            subtree.append(current)
            for child in tree_map.get(current, []):
                queue.append(child)
        return subtree

    def _joint_motion_span(self, joint: Dict) -> float:
        """Estimate motion span to pick a representative joint (avoid tiny first-part joints)."""
        jtype = joint.get('type')
        if jtype == 'continuous':
            return float(2.0 * np.pi)
        lower = float(joint.get('limit', {}).get('lower', -1.57))
        upper = float(joint.get('limit', {}).get('upper', 1.57))
        span = abs(upper - lower)
        # Some joints have near-zero ranges (e.g., buttons with 0.005). Avoid degeneracy.
        if not np.isfinite(span) or span < 1e-6:
            return float(np.pi / 6.0) if jtype in ['revolute'] else 0.01
        return float(span)

    def _select_joint_and_moving_links(
        self,
        urdf_data: Dict,
        part_meshes: Optional[Dict[str, trimesh.Trimesh]] = None,
        links: Optional[Dict[str, List[str]]] = None,
        obj_path: Optional[str] = None,
        joint_selection: str = "largest_motion",
        joint_idx: Optional[int] = None,
    ) -> Tuple[Dict, List[str]]:
        """
        Select a joint for sample generation and return the corresponding moving-subtree links.

        - joint_selection:
          - "first": the first movable joint
          - "random": a random movable joint
          - "largest_motion": maximize (motion span) × (moving geometry scale). Recommended default to
            avoid selecting tiny parts such as small buttons.
        """
        joints: List[Dict] = urdf_data.get('joints', [])
        if not joints:
            raise ValueError("No movable joints found")

        if joint_idx is not None:
            if joint_idx < 0 or joint_idx >= len(joints):
                raise IndexError(f"joint_idx {joint_idx} out of range (0..{len(joints)-1})")
            joint = joints[joint_idx]
            moving_links = self._get_subtree_links(joint['child'], urdf_data['full_tree_map'])
            return joint, moving_links

        if joint_selection == "first":
            joint = joints[0]
            moving_links = self._get_subtree_links(joint['child'], urdf_data['full_tree_map'])
            return joint, moving_links

        if joint_selection == "random":
            joint = joints[int(np.random.randint(0, len(joints)))]
            moving_links = self._get_subtree_links(joint['child'], urdf_data['full_tree_map'])
            return joint, moving_links

        # Default: "largest_motion"
        best_joint = None
        best_links: List[str] = []
        best_score = -1.0
        for j in joints:
            subtree_links = self._get_subtree_links(j['child'], urdf_data['full_tree_map'])
            # Estimate moving geometry scale (prefer vertex count; otherwise use mesh-file count as a proxy).
            geom_score = 0
            if part_meshes is not None:
                for ln in subtree_links:
                    mesh = part_meshes.get(ln)
                    if mesh is not None:
                        geom_score += int(len(mesh.vertices))
            else:
                links_dict = links
                if links_dict is None or obj_path is None:
                    geom_score = 0
                else:
                    for ln in subtree_links:
                        for fn in (links_dict.get(ln) or []):
                            if os.path.exists(os.path.join(obj_path, fn)):
                                geom_score += 1

            if geom_score <= 0:
                continue

            score = float(geom_score) * self._joint_motion_span(j)
            if score > best_score:
                best_score = score
                best_joint = j
                best_links = subtree_links

        if best_joint is None:
            # Fallback to "first".
            joint = joints[0]
            moving_links = self._get_subtree_links(joint['child'], urdf_data['full_tree_map'])
            return joint, moving_links

        return best_joint, best_links

    def load_part_meshes(self, obj_path: str, links: Dict[str, List[str]]) -> Dict[str, trimesh.Trimesh]:
        """
        Load meshes for each part/link.
        """
        part_meshes = {}

        for link_name, mesh_filenames in links.items():
            link_meshes = []
            for mesh_filename in mesh_filenames:
                mesh_path = os.path.join(obj_path, mesh_filename)
                if os.path.exists(mesh_path):
                    try:
                        mesh = trimesh.load(mesh_path, force='mesh', process=False)
                        link_meshes.append(mesh)
                    except Exception as e:
                        # print(f"Warning: Failed to load {mesh_path}: {e}")
                        pass

            if len(link_meshes) > 0:
                if len(link_meshes) == 1:
                    part_meshes[link_name] = link_meshes[0]
                else:
                    combined_mesh = trimesh.util.concatenate(link_meshes)
                    part_meshes[link_name] = combined_mesh

        return part_meshes

    def _load_mesh_safe(self, mesh_path: str) -> Optional[trimesh.Trimesh]:
        """
        More robust mesh loading:
        - Accept trimesh.Scene outputs
        - Keep process=False to avoid unnecessary recomputation
        """
        if not os.path.exists(mesh_path):
            return None
        try:
            m = trimesh.load(mesh_path, force='mesh', process=False)
        except Exception:
            return None

        if isinstance(m, trimesh.Scene):
            geoms = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                return None
            try:
                m = trimesh.util.concatenate(geoms)
            except Exception:
                return None

        if not isinstance(m, trimesh.Trimesh):
            return None
        if m.vertices is None or m.faces is None:
            return None
        if len(m.vertices) == 0 or len(m.faces) == 0:
            return None
        return m

    def _stream_sample_points(
        self,
        obj_path: str,
        links: Dict[str, List[str]],
        moving_links: set,
        num_points: int,
        joint_origin: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Low-memory mode: avoid mesh concatenation.
        Load each mesh file -> sample points -> release immediately.

        Returns:
          - points: [num_points, 3] float32
          - point_mask: [num_points] int32 (0=static, 1=movable)
          - movable_points: [K, 3] float32 (only for drag-point sampling; K<=num_points)
          - bounds_min: [3] float32
          - bounds_max: [3] float32
        """
        # Count available mesh files to allocate per-mesh sampling budget.
        mesh_entries: List[Tuple[str, int]] = []  # (mesh_path, is_moving)
        for link_name, mesh_filenames in links.items():
            is_moving = 1 if link_name in moving_links else 0
            for mesh_filename in mesh_filenames:
                mesh_path = os.path.join(obj_path, mesh_filename)
                if os.path.exists(mesh_path):
                    mesh_entries.append((mesh_path, is_moving))

        if not mesh_entries:
            raise ValueError(f"No meshes found under {obj_path}")

        n_total = len(mesh_entries)
        base = max(1, int(num_points // n_total))
        rem = int(num_points - base * n_total)
        if rem < 0:
            rem = 0

        all_points = []
        all_masks = []
        movable_points = []

        bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)

        # Sample points per mesh.
        for i, (mesh_path, is_moving) in enumerate(mesh_entries):
            n_i = base + (1 if i < rem else 0)
            if n_i <= 0:
                continue

            mesh = self._load_mesh_safe(mesh_path)
            if mesh is None:
                continue

            # Accumulate bounds.
            try:
                b = mesh.bounds.astype(np.float32)
                bounds_min = np.minimum(bounds_min, b[0])
                bounds_max = np.maximum(bounds_max, b[1])
            except Exception:
                pass

            try:
                pts, _ = trimesh.sample.sample_surface(mesh, n_i)
            except Exception:
                del mesh
                continue

            pts = pts.astype(np.float32, copy=False)
            mask = np.full((len(pts),), int(is_moving), dtype=np.int32)

            all_points.append(pts)
            all_masks.append(mask)
            if is_moving:
                movable_points.append(pts)

            # Proactively release large objects to reduce peak memory.
            del mesh
            if (i % 10) == 0:
                gc.collect()

        if not all_points:
            raise ValueError(f"Failed to sample any points from {obj_path}")

        points = np.concatenate(all_points, axis=0)
        point_mask = np.concatenate(all_masks, axis=0)

        # Trim/pad to exactly num_points (meshes may fail to load, causing mismatch).
        if len(points) >= num_points:
            points = points[:num_points]
            point_mask = point_mask[:num_points]
        else:
            # If insufficient, pad by resampling with replacement (avoid downstream failures).
            pad_n = num_points - len(points)
            rep_idx = np.random.randint(0, len(points), size=pad_n)
            points = np.concatenate([points, points[rep_idx]], axis=0)
            point_mask = np.concatenate([point_mask, point_mask[rep_idx]], axis=0)

        if movable_points:
            movable_points = np.concatenate(movable_points, axis=0)
        else:
            movable_points = points[point_mask.astype(bool)]

        if movable_points is None or len(movable_points) == 0:
            raise ValueError("No movable points sampled (moving subtree has no geometry?)")

        # If bounds were not updated (extreme edge cases), fall back to point-cloud bounds.
        if not np.isfinite(bounds_min).all() or not np.isfinite(bounds_max).all():
            bounds_min = points.min(axis=0).astype(np.float32)
            bounds_max = points.max(axis=0).astype(np.float32)

        return points, point_mask, movable_points, bounds_min, bounds_max

    def generate_training_sample(
        self,
        obj_idx: int,
        num_frames: int = 16,
        joint_selection: str = "largest_motion",
        joint_idx: Optional[int] = None,
        return_mesh: bool = True,
        num_points: Optional[int] = None,
    ) -> Dict:
        """
        Generate training sample with dual quaternion ground truth.
        Fix motion leakage by recursively collecting all descendant moving parts.
        """
        obj_path = self.object_list[obj_idx]

        # Parse URDF
        urdf_path = os.path.join(obj_path, "mobility_annotation_gapartnet.urdf")
        urdf_data = self.parse_urdf(urdf_path)

        if len(urdf_data['joints']) == 0:
            raise ValueError(f"No movable joints found in {obj_path}")

        # Select a joint for sample generation (default avoids always picking a tiny first joint).
        # When return_mesh=False, avoid loading full meshes (OOM); use mesh-file count as a proxy score.
        part_meshes = None
        if return_mesh:
            part_meshes = self.load_part_meshes(obj_path, urdf_data['links'])

        joint, moving_links_list = self._select_joint_and_moving_links(
            urdf_data=urdf_data,
            part_meshes=part_meshes,
            links=urdf_data.get('links'),
            obj_path=obj_path,
            joint_selection=joint_selection,
            joint_idx=joint_idx,
        )
        
        # --- Core: descendant propagation ---
        moving_root_link = joint['child']
        
        # Use the full tree map to collect all links that should move with this joint
        # (e.g., door -> handle, lock). `moving_root_link` may have many descendants via fixed joints.
        all_moving_links = set(moving_links_list if moving_links_list else self._get_subtree_links(moving_root_link, urdf_data['full_tree_map']))

        # ========== Low-memory point-cloud mode (for LMDB building) ==========
        if not return_mesh:
            if num_points is None:
                raise ValueError("return_mesh=False requires num_points")

            points, point_mask, movable_points, bounds_min, bounds_max = self._stream_sample_points(
                obj_path=obj_path,
                links=urdf_data['links'],
                moving_links=all_moving_links,
                num_points=int(num_points),
                joint_origin=joint['origin_xyz'],
            )

            # Sample a drag point: pick a movable point far from the joint origin.
            if joint['type'] in ['revolute', 'continuous']:
                d = np.linalg.norm(movable_points - joint['origin_xyz'][None, :], axis=1)
                if len(d) == 0:
                    raise ValueError("No movable points for drag sampling")
                k = max(1, len(d) // 5)  # top 20%
                far_idx = np.argpartition(d, -k)[-k:]
                drag_point_start = movable_points[np.random.choice(far_idx)].copy()

                lower, upper = joint['limit']['lower'], joint['limit']['upper']
                if upper - lower < np.pi:
                    lower, upper = -np.pi, np.pi
                angle_change = upper - lower
                rotation_direction = np.array(joint['axis']) * np.sign(angle_change)
            else:
                drag_point_start = movable_points[np.random.randint(0, len(movable_points))].copy()
                rotation_direction = None

            # Subsequent steps (DQ sequence / trajectory generation) follow the original logic.
            lower, upper = joint['limit']['lower'], joint['limit']['upper']

            if joint['type'] == 'continuous':
                upper = max(upper, np.pi)
                lower = min(lower, -np.pi)

            if upper <= lower:
                upper = lower + 1.0

            if upper > lower:
                angles = np.linspace(lower, upper, num_frames)
            else:
                if joint['type'] == 'continuous':
                    angles = np.linspace(-np.pi, np.pi, num_frames)
                else:
                    angles = np.linspace(0, np.pi/2, num_frames)

            qr_sequence = []
            qd_sequence = []
            for angle in angles:
                if joint['type'] == 'revolute' or joint['type'] == 'continuous':
                    qr, qd = axis_angle_to_dualquat(joint['axis'], joint['origin_xyz'], angle)
                elif joint['type'] == 'prismatic':
                    qr, qd = translation_to_dualquat(joint['axis'], angle)
                else:
                    raise ValueError(f"Unknown joint type: {joint['type']}")
                qr_sequence.append(qr)
                qd_sequence.append(qd)

            qr_sequence = np.array(qr_sequence)
            qd_sequence = np.array(qd_sequence)

            drag_trajectory = []
            for qr, qd in zip(qr_sequence, qd_sequence):
                point_rotated = quaternion_rotate(qr, drag_point_start)
                t_vec = 2.0 * quaternion_multiply(qd, quaternion_conjugate(qr))[1:]
                drag_trajectory.append(point_rotated + t_vec)
            drag_trajectory = np.array(drag_trajectory)

            if len(drag_trajectory) > 2:
                start_tangent = drag_trajectory[1] - drag_trajectory[0]
                mid_idx = len(drag_trajectory) // 2
                mid_tangent = drag_trajectory[mid_idx+1] - drag_trajectory[mid_idx]
                end_tangent = drag_trajectory[-1] - drag_trajectory[-2]
                trajectory_vectors = np.stack([start_tangent, mid_tangent, end_tangent])
            else:
                diff = drag_trajectory[-1] - drag_trajectory[0]
                trajectory_vectors = np.stack([diff, diff, diff])

            diffs = drag_trajectory[1:] - drag_trajectory[:-1]
            lengths = np.linalg.norm(diffs, axis=1)
            total_length = lengths.sum()
            if total_length > 0:
                avg_dir = np.mean(trajectory_vectors, axis=0)
                avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-8)
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
            rotation_sign = np.sign(np.dot(cross_axis_drag, relative_point)) or 1.0

            if rotation_direction is None:
                rotation_direction = np.array(joint['axis'])

            rotation_direction = rotation_direction * rotation_sign
            drag_vector = drag_vector * rotation_sign

            return {
                # Use ndarray instead of trimesh so upstream datasets can normalize/write LMDB directly.
                'initial_mesh': points.astype(np.float32, copy=False),
                'drag_point': drag_point_start.astype(np.float32, copy=False),
                'drag_vector': drag_vector.astype(np.float32, copy=False),
                'trajectory_vectors': trajectory_vectors.astype(np.float32, copy=False),
                'rotation_direction': rotation_direction.astype(np.float32, copy=False),
                'qr_sequence': qr_sequence.astype(np.float32, copy=False),
                'qd_sequence': qd_sequence.astype(np.float32, copy=False),
                'drag_trajectory': drag_trajectory.astype(np.float32, copy=False),
                'joint_type': joint['type'],
                # point-level mask (0=static, 1=movable)
                'part_mask': point_mask.astype(np.int32, copy=False),
                'obj_path': obj_path,
                'joint_axis': joint['axis'],
                'joint_origin': joint['origin_xyz'],
                # Optionally return bounds for upstream normalization.
                'bounds_min': bounds_min,
                'bounds_max': bounds_max,
            }

        # ========== Full mesh concatenation mode (for inference / visualization) ==========
        if part_meshes is None:
            part_meshes = self.load_part_meshes(obj_path, urdf_data['links'])
        # Gather moving meshes and static meshes.
        movable_meshes_list = []
        static_meshes_list = []

        for link_name, mesh in part_meshes.items():
            if link_name in all_moving_links:
                movable_meshes_list.append(mesh)
            else:
                static_meshes_list.append(mesh)

        if not movable_meshes_list:
            raise ValueError(f"Child link {moving_root_link} (and descendants) have no meshes")

        # Concatenate all moving parts.
        child_mesh = trimesh.util.concatenate(movable_meshes_list)

        # Sample a drag point.
        if joint['type'] in ['revolute', 'continuous']:
            # Sample the farthest point over the whole moving part.
            distances = np.linalg.norm(child_mesh.vertices - joint['origin_xyz'], axis=1)
            far_indices = np.argsort(distances)[-max(1, len(child_mesh.vertices) // 5):]  # Top 20%
            drag_point_idx = np.random.choice(far_indices)
            drag_point_start = child_mesh.vertices[drag_point_idx].copy()

            # Improve angle range
            lower, upper = joint['limit']['lower'], joint['limit']['upper']
            if upper - lower < np.pi:
                lower, upper = -np.pi, np.pi
            
            # For rotation_direction
            angle_change = upper - lower
            rotation_direction = np.array(joint['axis']) * np.sign(angle_change)

        else:
            drag_point_idx = np.random.randint(0, len(child_mesh.vertices))
            drag_point_start = child_mesh.vertices[drag_point_idx].copy()

        # Generate dual quaternion sequence
        lower, upper = joint['limit']['lower'], joint['limit']['upper']

        if joint['type'] == 'continuous':
            upper = max(upper, np.pi)  
            lower = min(lower, -np.pi)
        
        # Protect against weird limits
        if upper <= lower:
             upper = lower + 1.0

        if upper > lower:
            angles = np.linspace(lower, upper, num_frames)
        else:
            if joint['type'] == 'continuous':
                angles = np.linspace(-np.pi, np.pi, num_frames)
            else:
                angles = np.linspace(0, np.pi/2, num_frames)  

        qr_sequence = []
        qd_sequence = []

        for angle in angles:
            if joint['type'] == 'revolute' or joint['type'] == 'continuous':
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

        # Combine all parts into initial mesh (Static + Combined Movable)
        if static_meshes_list:
            static_mesh_combined = trimesh.util.concatenate(static_meshes_list)
            initial_mesh = trimesh.util.concatenate([static_mesh_combined, child_mesh])
            
            # Create part mask (0=static, 1=movable)
            num_static_verts = len(static_mesh_combined.vertices)
            num_movable_verts = len(child_mesh.vertices)
            part_mask = np.concatenate([
                np.zeros(num_static_verts, dtype=np.int32),
                np.ones(num_movable_verts, dtype=np.int32)
            ])
        else:
            # Only the moving part is present.
            initial_mesh = child_mesh
            part_mask = np.ones(len(child_mesh.vertices), dtype=np.int32)

        # Calculate drag vector with trajectory information
        drag_trajectory = []
        for qr, qd in zip(qr_sequence, qd_sequence):
            point_rotated = quaternion_rotate(qr, drag_point_start)
            t_vec = 2.0 * quaternion_multiply(qd, quaternion_conjugate(qr))[1:]
            point_transformed = point_rotated + t_vec
            drag_trajectory.append(point_transformed)

        drag_trajectory = np.array(drag_trajectory)  # [num_frames, 3]

        # Multi-tangent
        if len(drag_trajectory) > 2:
            start_tangent = drag_trajectory[1] - drag_trajectory[0]
            mid_idx = len(drag_trajectory) // 2
            mid_tangent = drag_trajectory[mid_idx+1] - drag_trajectory[mid_idx]
            end_tangent = drag_trajectory[-1] - drag_trajectory[-2]
            trajectory_vectors = np.stack([start_tangent, mid_tangent, end_tangent])
        else:
            # Fallback for very short trajectories
            diff = drag_trajectory[-1] - drag_trajectory[0]
            trajectory_vectors = np.stack([diff, diff, diff]) # Simple broadcast

        # Improved drag_vector
        diffs = drag_trajectory[1:] - drag_trajectory[:-1]
        lengths = np.linalg.norm(diffs, axis=1)
        total_length = lengths.sum()
        if total_length > 0:
            avg_dir = np.mean(trajectory_vectors, axis=0)
            avg_dir = avg_dir / (np.linalg.norm(avg_dir) + 1e-8)
            drag_vector = avg_dir * total_length
        else:
            drag_vector = drag_trajectory[-1] - drag_trajectory[0]

        # Rotation Direction & Sign Fix
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
        rotation_sign = np.sign(np.dot(cross_axis_drag, relative_point)) or 1.0

        if rotation_direction is None:
             rotation_direction = np.array(joint['axis'])

        rotation_direction = rotation_direction * rotation_sign
        drag_vector = drag_vector * rotation_sign

        return {
            'initial_mesh': initial_mesh,
            'drag_point': drag_point_start,
            'drag_vector': drag_vector,
            'trajectory_vectors': trajectory_vectors,  
            'rotation_direction': rotation_direction,  
            'qr_sequence': qr_sequence,
            'qd_sequence': qd_sequence,
            'drag_trajectory': drag_trajectory,  # extra field for LMDB checks
            'joint_type': joint['type'],
            'part_mask': part_mask,
            'obj_path': obj_path,
            'joint_axis': joint['axis'],  
            'joint_origin': joint['origin_xyz']  
        }

    def __len__(self):
        return len(self.object_list)


# ==============================================================================
#  DragMeshDatasetV2 (PyTorch Dataset)
# ==============================================================================

class DragMeshDatasetV2(torch.utils.data.Dataset):
    """
    PyTorch Dataset with Dual Quaternion ground truth.
    """

    def __init__(self, dataset_root: str, num_frames: int = 16, num_points: int = 4096, joint_selection: str = "largest_motion"):
        self.loader = GAPartNetLoaderV2(dataset_root)
        self.num_frames = num_frames
        self.num_points = num_points
        self.joint_selection = joint_selection

    def mesh_to_pointcloud(self, mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
        """Sample point cloud from mesh surface."""
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        return points

    def normalize_mesh(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
        """
        Normalize mesh to unit bounding box centered at origin.
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
        sample = self.loader.generate_training_sample(idx, self.num_frames, joint_selection=self.joint_selection)

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
            
        # Propagate drag_trajectory if needed.
        if 'drag_trajectory' in sample:
            traj = (sample['drag_trajectory'] - center) / scale
            result['drag_trajectory'] = torch.from_numpy(traj).float()

        return result