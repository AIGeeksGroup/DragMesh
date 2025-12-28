"""
inference_animation_kpp.py

This script mirrors the inference/export logic in `inference_animation.py` and optionally
overrides joint parameters using a Keypoint Predictor (KPP): joint_type / joint_axis / joint_origin.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader_v2 import GAPartNetLoaderV2
from modules.predictor import KeypointPredictor

# Reuse the canonical animation logic (single-interaction trajectory, loop_mode, headless rendering
# fallback, and animated GLB injection).
from inference_animation import FixedGAPartNetLoader, load_model, run_animation_from_sample


def load_kpp_model(checkpoint_path: str, device: torch.device) -> Optional[KeypointPredictor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as exc:
        print(f"Error: unable to load KPP checkpoint: {exc}")
        return None
    config = checkpoint.get('config', {})
    kpp_model = KeypointPredictor(
        use_mask=config.get('use_mask', True),
        use_drag=config.get('use_drag', True)
    ).to(device)
    state = {k.replace('module.', ''): v for k, v in checkpoint.get('model_state_dict', {}).items()}
    try:
        kpp_model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        print(f"[WARN] KPP strict load failed, fallback to strict=False: {exc}")
        kpp_model.load_state_dict(state, strict=False)
    kpp_model.eval()
    return kpp_model


def _prepare_kpp_inputs_from_mesh(initial_mesh: trimesh.Trimesh,
                                  vertex_part_mask: np.ndarray,
                                  drag_point_world: np.ndarray,
                                  drag_vector_world: np.ndarray,
                                  num_points: int = 4096) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Match the normalization protocol used by `inference_animation.py`:
    - Normalize the mesh with bbox center/scale
    - Sample a surface point cloud with `num_points`
    - Derive per-sampled-point mask via face -> vertex majority vote (0/1)
    - Normalize drag point/vector with the same center/scale
    """
    bounds = initial_mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = (bounds[1] - bounds[0]).max()
    if scale < 1e-6:
        scale = 1.0

    mesh_norm = initial_mesh.copy()
    mesh_norm.vertices = (mesh_norm.vertices - center) / scale

    pc, face_indices = trimesh.sample.sample_surface(mesh_norm, num_points)

    sampled_mask = np.zeros((len(face_indices),), dtype=np.float32)
    # vertex_part_mask is per-vertex (bool/0/1). For each sampled point, use majority vote on its face.
    for i, fid in enumerate(face_indices):
        face_vertices = initial_mesh.faces[int(fid)]
        face_mask_values = vertex_part_mask[face_vertices].astype(np.int32)
        sampled_mask[i] = float(np.bincount(face_mask_values).argmax())

    drag_point_norm = (np.asarray(drag_point_world) - center) / scale
    drag_vector_norm = np.asarray(drag_vector_world) / scale
    return pc, sampled_mask, drag_point_norm, drag_vector_norm, center, scale


def run_kpp_animation(model,
                      loader: GAPartNetLoaderV2,
                      sample_idx: int,
                      device: torch.device,
                      output_dir: str,
                      num_frames: int,
                      num_samples_to_gen: int,
                      kpp_model: Optional[KeypointPredictor],
                      force_rotation: bool,
                      fps: float,
                      loop_mode: str,
                      use_kpp_type: bool = True):
    sample = loader.generate_training_sample(sample_idx, num_frames=num_frames)
    if sample is None:
        print(f"Error: unable to generate a valid sample for index={sample_idx}.")
        return

    initial_mesh = sample['initial_mesh']
    vertex_part_mask = np.asarray(sample['part_mask'])
    if vertex_part_mask.dtype != bool:
        vertex_part_mask = vertex_part_mask.astype(bool)

    # Default: use GT joint parameters (same behavior as inference_animation.py).
    joint_type_str = sample['joint_type']
    joint_axis = np.asarray(sample['joint_axis'], dtype=np.float32)
    joint_origin = np.asarray(sample['joint_origin'], dtype=np.float32)
    joint_type_int = 0 if joint_type_str in ['revolute', 'continuous'] else 1

    # If KPP is provided: override joint parameters with predictions (the only difference vs animation).
    if kpp_model is not None:
        pc, sampled_mask, drag_point_norm, drag_vector_norm, center, scale = _prepare_kpp_inputs_from_mesh(
            initial_mesh=initial_mesh,
            vertex_part_mask=vertex_part_mask.astype(np.int32),
            drag_point_world=np.asarray(sample['drag_point']),
            drag_vector_world=np.asarray(sample['drag_vector']),
            num_points=4096
        )

        with torch.no_grad():
            pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(sampled_mask).float().unsqueeze(0).to(device)
            drag_point_tensor = torch.from_numpy(drag_point_norm).float().unsqueeze(0).to(device)
            drag_vector_tensor = torch.from_numpy(drag_vector_norm).float().unsqueeze(0).to(device)
            pred_type_logits, pred_axis, pred_origin = kpp_model(
                pc_tensor, mask_tensor, drag_point_tensor, drag_vector_tensor
            )

        pred_axis_np = pred_axis.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_axis_np = pred_axis_np / (np.linalg.norm(pred_axis_np) + 1e-8)
        pred_origin_norm = pred_origin.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_origin_world = pred_origin_norm * scale + center

        joint_axis = pred_axis_np
        joint_origin = pred_origin_world
        if use_kpp_type and pred_type_logits is not None:
            joint_type_int = int(torch.argmax(pred_type_logits, dim=-1).item())
            joint_type_str = 'revolute' if joint_type_int == 0 else 'prismatic'

        print(f"[KPP] joint_type={joint_type_str}, axis={joint_axis.tolist()}, origin={joint_origin.tolist()}")

    if force_rotation:
        joint_type_int = 0
        joint_type_str = 'revolute'
        print("*** Forcing rotation mode (joint_type=0) ***")

    chosen_id = os.path.basename(loader.object_list[sample_idx])
    output_dir_id = os.path.join(output_dir, chosen_id)
    os.makedirs(output_dir_id, exist_ok=True)

    # Build an "animation-compatible" sample dict and reuse run_animation_from_sample().
    sample_for_anim = {
        'initial_mesh': initial_mesh,
        'part_mask': vertex_part_mask,
        'drag_point': np.asarray(sample['drag_point']),
        'drag_vector': np.asarray(sample['drag_vector']),
        'joint_type': joint_type_str,
        'joint_axis': joint_axis,
        'joint_origin': joint_origin,
        # Optional conditioning fields (keep consistent with training if available).
        'rotation_direction': sample.get('rotation_direction'),
        'trajectory_vectors': sample.get('trajectory_vectors'),
        'drag_trajectory': sample.get('drag_trajectory'),
    }

    run_animation_from_sample(
        model=model,
        sample=sample_for_anim,
        sample_name=chosen_id,
        device=device,
        output_root=output_dir,
        num_frames=num_frames,
        num_samples_to_gen=num_samples_to_gen,
        force_rotation=False,  # already handled above
        include_groundtruth=False,
        fps=fps,
        loop_mode=loop_mode
    )


def parse_args():
    parser = argparse.ArgumentParser(description='VAE Animation Inference (optionally with KPP joint prediction)')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='VAE checkpoint')
    parser.add_argument('--sample_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_animation_kpp')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--force_rotation', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--transformer_heads', type=int, default=8)

    parser.add_argument('--kpp_checkpoint', type=str, default=None, help='Path to KPP weights (optional)')
    parser.add_argument('--use_kpp_type', action='store_true', default=True, help='Use KPP predicted joint type (default: True)')
    parser.add_argument('--no_use_kpp_type', action='store_false', dest='use_kpp_type',
                        help='Do not override joint type; only use KPP axis/origin.')

    parser.add_argument('--fps', type=float, default=5.0, help='animation playback fps (smaller = slower)')
    parser.add_argument('--loop_mode', type=str, default='pingpong', choices=['once', 'pingpong'],
                        help='once: 0->1; pingpong: 0->1->0 (better for default looping players)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loader = FixedGAPartNetLoader(dataset_root=args.dataset_root)
    if len(loader.object_list) == 0:
        print("Error: empty dataset.")
        return

    ids_list = [os.path.basename(p) for p in loader.object_list]
    if args.sample_id not in ids_list:
        print(f"ERROR: Sample ID '{args.sample_id}' not found.")
        return

    idx = ids_list.index(args.sample_id)

    vae_model, actual_num_frames = load_model(args.checkpoint, device, args)
    if vae_model is None:
        return

    num_frames_to_gen = args.num_frames if args.num_frames is not None else actual_num_frames

    kpp_model = None
    if args.kpp_checkpoint:
        kpp_model = load_kpp_model(args.kpp_checkpoint, device=device)

    run_kpp_animation(
        model=vae_model,
        loader=loader,
        sample_idx=idx,
        device=device,
        output_dir=args.output_dir,
        num_frames=num_frames_to_gen,
        num_samples_to_gen=args.num_samples,
        kpp_model=kpp_model,
        force_rotation=args.force_rotation,
        fps=args.fps,
        loop_mode=args.loop_mode,
        use_kpp_type=args.use_kpp_type
    )


if __name__ == '__main__':
    main()
