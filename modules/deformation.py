"""
Mesh Deformation Module using ArtGS's Dual Quaternion

This module applies predicted dual quaternion transformations to mesh vertices
to generate animated mesh sequences.
"""

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from typing import Tuple, List, Dict
from modules.dual_quaternion import (
    dual_quaternion_apply,
    dual_quaternion_to_quaternion_translation,
    quaternion_translation_apply
)


class MeshDeformationEngine:
    """
    Apply dual quaternion transformations to mesh vertices.
    Uses ArtGS's dual_quaternion_apply for smooth articulated motion.
    """

    def __init__(self, device='cpu'):
        self.device = device

    def apply_dualquat_to_vertices(
        self,
        vertices: torch.Tensor,
        qr: torch.Tensor,
        qd: torch.Tensor,
        part_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply dual quaternion transformation to vertices.

        Args:
            vertices: Vertex positions [N, 3] or [B, N, 3]
            qr: Rotation quaternion [4] or [B, 4]
            qd: Translation quaternion [4] or [B, 4]
            part_mask: Which vertices to transform [N] (0=static, 1=movable)
                      If None, transform all vertices

        Returns:
            Transformed vertices [N, 3] or [B, N, 3]
        """
        # Handle batched vs single input
        is_batched = vertices.dim() == 3

        if part_mask is None:
            # Transform all vertices
            deformed_vertices = dual_quaternion_apply((qr, qd), vertices)
        else:
            # Only transform movable vertices
            if is_batched:
                # Batched processing
                B, N, _ = vertices.shape
                deformed_vertices = vertices.clone()
                movable_mask = (part_mask == 1)

                for b in range(B):
                    movable_verts = vertices[b, movable_mask]  # [M, 3]
                    transformed = dual_quaternion_apply((qr[b], qd[b]), movable_verts)
                    deformed_vertices[b, movable_mask] = transformed
            else:
                # Single mesh processing
                deformed_vertices = vertices.clone()
                movable_mask = (part_mask == 1)
                movable_verts = vertices[movable_mask]  # [M, 3]
                transformed = dual_quaternion_apply((qr, qd), movable_verts)
                deformed_vertices[movable_mask] = transformed

        return deformed_vertices

    def generate_mesh_sequence(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        qr_sequence: torch.Tensor,
        qd_sequence: torch.Tensor,
        part_mask: torch.Tensor = None
    ) -> List[trimesh.Trimesh]:
        """
        Generate animated mesh sequence from dual quaternion sequence.

        Args:
            vertices: Initial vertex positions [N, 3]
            faces: Face indices [F, 3]
            qr_sequence: Rotation quaternion sequence [T, 4]
            qd_sequence: Translation quaternion sequence [T, 4]
            part_mask: Which vertices are movable [N]

        Returns:
            List of trimesh objects [T frames]
        """
        num_frames = qr_sequence.shape[0]
        mesh_sequence = []

        vertices = vertices.to(self.device)
        qr_sequence = qr_sequence.to(self.device)
        qd_sequence = qd_sequence.to(self.device)
        if part_mask is not None:
            part_mask = part_mask.to(self.device)

        for t in range(num_frames):
            # Apply transformation for frame t
            deformed_verts = self.apply_dualquat_to_vertices(
                vertices,
                qr_sequence[t],
                qd_sequence[t],
                part_mask
            )

            # Convert to numpy and create trimesh
            verts_np = deformed_verts.cpu().numpy()
            faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

            mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
            mesh_sequence.append(mesh)

        return mesh_sequence

    def deform_mesh_with_vae_prediction(
        self,
        mesh: trimesh.Trimesh,
        qr_pred: torch.Tensor,
        qd_pred: torch.Tensor,
        part_mask: np.ndarray = None
    ) -> List[trimesh.Trimesh]:
        """
        Deform mesh using VAE predictions.

        Args:
            mesh: Initial mesh (trimesh object)
            qr_pred: Predicted rotation quaternions [T, 4]
            qd_pred: Predicted translation quaternions [T, 4]
            part_mask: Which vertices are movable [N] (numpy array)

        Returns:
            List of deformed meshes [T frames]
        """
        # Convert mesh to torch tensors
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()

        if part_mask is not None:
            part_mask = torch.from_numpy(part_mask).long()

        # Generate sequence
        mesh_sequence = self.generate_mesh_sequence(
            vertices, faces, qr_pred, qd_pred, part_mask
        )

        return mesh_sequence


def visualize_deformation(
    mesh_sequence: List[trimesh.Trimesh],
    output_dir: str,
    prefix: str = "frame"
):
    """
    Save mesh sequence as individual OBJ files.

    Args:
        mesh_sequence: List of trimesh objects
        output_dir: Directory to save meshes
        prefix: Prefix for filenames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, mesh in enumerate(mesh_sequence):
        output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.obj")
        mesh.export(output_path)

    print(f"Saved {len(mesh_sequence)} frames to {output_dir}")


def create_animation_video(
    mesh_sequence: List[trimesh.Trimesh],
    output_path: str,
    resolution=(800, 600),
    fps=10
):
    """
    Create video from mesh sequence.

    Args:
        mesh_sequence: List of trimesh objects
        output_path: Path to save video (e.g., 'animation.mp4')
        resolution: Video resolution (width, height)
        fps: Frames per second
    """
    try:
        import pyrender
        import imageio
        import tempfile
    except ImportError:
        print("Warning: pyrender or imageio not installed. Cannot create video.")
        print("Install with: pip install pyrender imageio[ffmpeg]")
        return

    # Create pyrender scene
    scene = pyrender.Scene()

    # Add camera and light
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render frames
    renderer = pyrender.OffscreenRenderer(*resolution)
    frames = []

    for i, mesh in enumerate(mesh_sequence):
        # Convert trimesh to pyrender mesh
        mesh_pr = pyrender.Mesh.from_trimesh(mesh)

        # Add mesh to scene
        mesh_node = scene.add(mesh_pr)

        # Render
        color, _ = renderer.render(scene)
        frames.append(color)

        # Remove mesh for next frame
        scene.remove_node(mesh_node)

    # Save video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved animation to {output_path}")

    renderer.delete()


# Example usage
if __name__ == "__main__":
    print("Testing Mesh Deformation Engine...")

    # Create a simple test mesh (cube)
    vertices = torch.tensor([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Front
        [4, 5, 6], [4, 6, 7],  # Back
        [0, 1, 5], [0, 5, 4],  # Bottom
        [2, 3, 7], [2, 7, 6],  # Top
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ])

    # Part mask: top half is movable
    part_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    # Create a simple rotation sequence
    num_frames = 16
    angles = torch.linspace(0, np.pi / 2, num_frames)

    qr_sequence = []
    qd_sequence = []

    for angle in angles:
        # Rotation around Y axis
        qr = torch.tensor([
            np.cos(angle / 2),
            0,
            np.sin(angle / 2),
            0
        ], dtype=torch.float32)

        # No translation
        qd = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        qr_sequence.append(qr)
        qd_sequence.append(qd)

    qr_sequence = torch.stack(qr_sequence)
    qd_sequence = torch.stack(qd_sequence)

    # Create deformation engine
    engine = MeshDeformationEngine()

    # Generate mesh sequence
    print("\nGenerating mesh sequence...")
    mesh_sequence = engine.generate_mesh_sequence(
        vertices, faces, qr_sequence, qd_sequence, part_mask
    )

    print(f"Generated {len(mesh_sequence)} frames")
    print(f"First frame vertices: {mesh_sequence[0].vertices.shape}")
    print(f"Last frame vertices: {mesh_sequence[-1].vertices.shape}")

    # Visualize deformation
    print("\nSaving frames...")
    visualize_deformation(mesh_sequence, "outputs/test_deformation", "test_cube")

    print("\nTest complete!")
