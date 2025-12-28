# -------------------------------------------------------------------
# inference_animation.py
# -------------------------------------------------------------------

import sys
import os
import torch
import numpy as np
import trimesh
import argparse
from tqdm import tqdm
from difflib import get_close_matches
import tempfile 
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch.nn.functional as F
import math

from modules.model_v2 import DualQuaternionVAE 
from modules.data_loader_v2 import GAPartNetLoaderV2
from modules.dual_quaternion import dual_quaternion_apply, quaternion_to_axis_angle, quaternion_multiply, quaternion_conjugate

# --- Offscreen rendering (headless fallback) ---
_OFFSCREEN_RENDERING_BROKEN = False

def _maybe_set_headless_opengl_platform():
    """
    In headless environments (no DISPLAY), `pyrender` may default to `pyglet` and raise
    `NoSuchDisplayException`. We set `PYOPENGL_PLATFORM` before importing pyrender to
    prefer EGL (GPU) and fall back to OSMesa (software).
    """
    if os.environ.get("DISPLAY"):
        return
    if os.environ.get("PYOPENGL_PLATFORM"):
        return
    # Prefer EGL (common on GPU servers); fall back to OSMesa (software rendering).
    os.environ["PYOPENGL_PLATFORM"] = "egl"

# --- glTF animation support ---
try:
    import pygltflib
    from pygltflib import GLTF2
except ImportError:
    print("\n--- Warning: pygltflib not installed ---")
    print("Skipping animated GLB export. Install with: pip install pygltflib")
    pygltflib = None

# --- Helpers ---
def project_rotation_to_axis(quaternion, target_axis):
    """Project a quaternion to a target axis (PyTorch tensor implementation)."""
    target_axis = target_axis.to(quaternion.device)
    angle = 2 * torch.acos(torch.clamp(quaternion[..., 0:1].abs(), -1.0, 1.0))
    target_axis_normalized = F.normalize(target_axis, p=2, dim=-1)
    half_angle = angle / 2.0
    projected_quaternion = torch.cat([
        torch.cos(half_angle),
        target_axis_normalized * torch.sin(half_angle)
    ], dim=-1)
    projected_quaternion = F.normalize(projected_quaternion, p=2, dim=-1)
    return projected_quaternion

def _nlerp_quaternion_sequence(q_start: torch.Tensor, q_end: torch.Tensor, t_values: torch.Tensor) -> torch.Tensor:
    """
    Normalized linear interpolation (NLERP) for quaternions.

    This is used to produce a "single interaction" rotation trajectory (no cumulative spinning).
    Args:
        q_start/q_end: [4] (wxyz)
        t_values: [T]
    Returns:
        [T, 4]
    """
    # Handle double cover: always take the shortest arc.
    dot = torch.dot(q_start, q_end)
    if dot < 0.0:
        q_end = -q_end
    q = (1.0 - t_values.unsqueeze(-1)) * q_start.unsqueeze(0) + t_values.unsqueeze(-1) * q_end.unsqueeze(0)
    return F.normalize(q, p=2, dim=-1)

def _make_loop_t_values(num_frames: int, loop_mode: str, device: torch.device) -> torch.Tensor:
    """
    loop_mode:
      - once: 0->1
      - pingpong: 0->1->0 (seamless looping in common viewers; avoids continuous spinning)
    """
    if num_frames <= 1:
        return torch.zeros((num_frames,), device=device)
    if loop_mode == "pingpong":
        forward_steps = num_frames // 2 + 1
        backward_steps = num_frames - forward_steps
        t_fwd = torch.linspace(0.0, 1.0, steps=forward_steps, device=device)
        if backward_steps <= 0:
            return t_fwd
        t_bwd_full = torch.linspace(1.0, 0.0, steps=backward_steps + 1, device=device)
        t_bwd = t_bwd_full[1:]  # drop the duplicated 1.0
        return torch.cat([t_fwd, t_bwd], dim=0)
    # default: once
    return torch.linspace(0.0, 1.0, steps=num_frames, device=device)

# --- Rendering 1: GIF ---
def create_gif(mesh_sequence, output_path, resolution=(600, 600), fps=15):
    global _OFFSCREEN_RENDERING_BROKEN
    if _OFFSCREEN_RENDERING_BROKEN:
        return
    try:
        _maybe_set_headless_opengl_platform()
        import pyrender
        import imageio
    except ImportError:
        return
    except Exception as e:
        _OFFSCREEN_RENDERING_BROKEN = True
        print(f"[WARN] Offscreen renderer init failed (headless?): {type(e).__name__}: {e}")
        print("       Tips: (1) export GLB (install pygltflib), (2) set PYOPENGL_PLATFORM=egl/osmesa, "
              "(3) run on a machine with a display.")
        return

    print(f"Generating GIF: {output_path}")
    if not mesh_sequence: return
    renderer = None
    try:
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[255, 255, 255])
        first_mesh = mesh_sequence[0]
        
        camera_pose = np.eye(4)
        zoom = np.max(first_mesh.extents) * 2.5
        camera_pose[2, 3] = zoom

        angle_y = math.radians(30)
        R_y = np.array([[math.cos(angle_y), 0, math.sin(angle_y), 0], [0, 1, 0, 0], [-math.sin(angle_y), 0, math.cos(angle_y), 0], [0, 0, 0, 1]])
        angle_x = math.radians(-30)
        R_x = np.array([[1, 0, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x), 0], [0, math.sin(angle_x), math.cos(angle_x), 0], [0, 0, 0, 1]])
        camera_pose = R_x @ R_y @ camera_pose

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0), pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(*resolution)
        frames = []

        for mesh in mesh_sequence:
            render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            mesh_node = scene.add(render_mesh)
            color, _ = renderer.render(scene)
            frames.append(color)
            scene.remove_node(mesh_node)

        imageio.mimsave(output_path, frames, fps=fps)
    except Exception as e:
        _OFFSCREEN_RENDERING_BROKEN = True
        print(f"[WARN] GIF render failed, skipping rendering. {type(e).__name__}: {e}")
    finally:
        try:
            if renderer is not None:
                renderer.delete()
        except Exception:
            pass

# --- Rendering 2: MP4 ---
def create_animation_video_local(mesh_sequence, output_path, resolution=(600, 600), fps=15):
    global _OFFSCREEN_RENDERING_BROKEN
    if _OFFSCREEN_RENDERING_BROKEN:
        return
    try:
        _maybe_set_headless_opengl_platform()
        import pyrender
        import imageio
    except ImportError:
        return
    except Exception as e:
        _OFFSCREEN_RENDERING_BROKEN = True
        print(f"[WARN] Offscreen renderer init failed (headless?): {type(e).__name__}: {e}")
        print("       Tips: (1) export GLB (install pygltflib), (2) set PYOPENGL_PLATFORM=egl/osmesa, "
              "(3) run on a machine with a display.")
        return

    print(f"Generating MP4: {output_path}")
    if not mesh_sequence: return
    renderer = None
    try:
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.3], bg_color=[255, 255, 255])
        first_mesh = mesh_sequence[0]
        
        camera_pose = np.eye(4)
        zoom = np.max(first_mesh.extents) * 2.5
        camera_pose[2, 3] = zoom

        angle_y = math.radians(30)
        R_y = np.array([[math.cos(angle_y), 0, math.sin(angle_y), 0], [0, 1, 0, 0], [-math.sin(angle_y), 0, math.cos(angle_y), 0], [0, 0, 0, 1]])
        angle_x = math.radians(-30)
        R_x = np.array([[1, 0, 0, 0], [0, math.cos(angle_x), -math.sin(angle_x), 0], [0, math.sin(angle_x), math.cos(angle_x), 0], [0, 0, 0, 1]])
        camera_pose = R_x @ R_y @ camera_pose

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0), pose=camera_pose)
        renderer = pyrender.OffscreenRenderer(*resolution)
        frames = []

        for mesh in mesh_sequence:
            render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            mesh_node = scene.add(render_mesh)
            color, _ = renderer.render(scene)
            frames.append(color)
            scene.remove_node(mesh_node)

        # MP4 export relies on the imageio-ffmpeg backend; keep this best-effort (do not fail inference).
        imageio.mimsave(output_path, frames, fps=fps)
    except Exception as e:
        _OFFSCREEN_RENDERING_BROKEN = True
        print(f"[WARN] MP4 render failed, skipping rendering. {type(e).__name__}: {e}")
        print("       Continuing inference outputs. For video/GIF, run with DISPLAY or configure EGL/OSMesa.")
    finally:
        try:
            if renderer is not None:
                renderer.delete()
        except Exception:
            pass


# --- Core: export an animated GLB ---
def export_animated_glb(initial_mesh, part_mask_np, qr_seq_tensor, qd_seq_tensor, 
                        joint_origin_norm_tensor, scale, center, output_path, fps=10):
    """
    Export an animated GLB with translation/rotation tracks.
    """
    if pygltflib is None:
        return

    print(f"Generating Animated GLB: {output_path}")

    def _pad_to_4bytes(blob: bytearray) -> None:
        """glTF bufferView requires 4-byte alignment (GLB viewers often rely on this)."""
        pad = (-len(blob)) % 4
        if pad:
            blob.extend(b"\x00" * pad)

    # 1) De-normalize joint origin back to world space.
    real_origin = joint_origin_norm_tensor.cpu().numpy() * scale + center
    
    # 2) Split the mesh into static vs moving parts (preserve materials via trimesh submesh).
    try:
        # All faces
        faces = initial_mesh.faces
        vertex_mask = part_mask_np.astype(bool)
        
        # Face assignment: a face is moving iff all its vertices are marked movable.
        # This heuristic works well for rigid part segmentations (e.g., GAPartNet).
        face_mask = vertex_mask[faces].all(axis=1)

        moving_face_ids = np.where(face_mask)[0]
        static_face_ids = np.where(~face_mask)[0]

        # Degenerate masks (all-moving or all-static): avoid empty submeshes.
        if moving_face_ids.size == 0 or static_face_ids.size == 0:
            initial_mesh.export(output_path)
            return

        # trimesh.submesh expects face index lists, not boolean masks.
        mesh_moving = initial_mesh.submesh([moving_face_ids], append=True)
        mesh_static = initial_mesh.submesh([static_face_ids], append=True)
        
        # Name nodes for debugging.
        mesh_moving.name = "moving_part"
        mesh_static.name = "static_part"
        
    except Exception as e:
        print(f"Error splitting mesh for GLB: {e}")
        # Fallback: export a static GLB if splitting fails.
        initial_mesh.export(output_path)
        return

    # 3) Pivot adjustment:
    #    A) translate moving vertices so the joint origin becomes (0, 0, 0) in its local frame
    #    B) set the moving node translation back to the joint origin in world space
    
    mesh_moving.vertices -= real_origin
    
    # 4) Build a trimesh.Scene and export a base GLB (trimesh handles material packaging).
    scene = trimesh.Scene()
    scene.add_geometry(mesh_static, node_name='static_node')
    
    # Initial transform: translate the moving node to the joint pivot.
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = real_origin
    scene.add_geometry(mesh_moving, node_name='moving_node', transform=transform_matrix)
    
    # Export a temporary GLB (trimesh performs binary conversion and material handling).
    fd, temp_glb_path = tempfile.mkstemp(suffix='.glb')
    os.close(fd)
    scene.export(temp_glb_path)
    
    # 5) Inject animation data via pygltflib.
    gltf = GLTF2().load(temp_glb_path)
    
    # Locate the moving node index.
    moving_node_idx = -1
    for i, node in enumerate(gltf.nodes):
        if node.name == 'moving_node':
            moving_node_idx = i
            break
    
    # Fallback: trimesh may rename nodes; the last node is typically the last added.
    if moving_node_idx == -1:
         moving_node_idx = len(gltf.nodes) - 1

    # --- Keyframe data ---
    num_frames = qr_seq_tensor.shape[0]
    times = (np.arange(num_frames, dtype=np.float32) / float(fps)).astype(np.float32)

    # Force TRS on the moving node. Some exporters write node.matrix, which can cause
    # translation/rotation tracks to be ignored in certain viewers.
    try:
        node = gltf.nodes[moving_node_idx]
        if getattr(node, "matrix", None):
            node.matrix = None
        node.translation = real_origin.astype(np.float32).tolist()
        if node.rotation is None:
            node.rotation = [0.0, 0.0, 0.0, 1.0]
        if node.scale is None:
            node.scale = [1.0, 1.0, 1.0]
    except Exception:
        pass
    
    # A) Rotation: DualQuaternion real part -> glTF quaternion.
    # PyTorch: [w, x, y, z] -> glTF: [x, y, z, w]
    qr_seq = qr_seq_tensor.detach().cpu().numpy().astype(np.float32)
    # Normalize to avoid non-unit quaternions (some viewers reject them).
    qr_norm = np.linalg.norm(qr_seq, axis=1, keepdims=True)
    qr_norm = np.where(qr_norm < 1e-8, 1.0, qr_norm)
    qr_seq = qr_seq / qr_norm
    rotations = np.zeros((num_frames, 4), dtype=np.float32)
    rotations[:, 0] = qr_seq[:, 1] # x
    rotations[:, 1] = qr_seq[:, 2] # y
    rotations[:, 2] = qr_seq[:, 3] # z
    rotations[:, 3] = qr_seq[:, 0] # w
    
    # B) Translation:
    # The node is already positioned at `real_origin`. The DQ translation is an offset
    # in normalized space, so per-frame translation is: real_origin + t * scale.
    
    qr_conj = quaternion_conjugate(qr_seq_tensor)
    t_q = 2.0 * quaternion_multiply(qd_seq_tensor, qr_conj)
    t_vec_norm = t_q[:, 1:].cpu().numpy() # [N, 3]
    t_vec_real = t_vec_norm * scale 
    
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    for i in range(num_frames):
        translations[i] = real_origin + t_vec_real[i]

    # --- Write binary buffer ---
    blob = bytearray(gltf.binary_blob())
    _pad_to_4bytes(blob)
    
    # 1. Times Input
    times_bytes = times.tobytes()
    times_offset = len(blob)
    blob.extend(times_bytes)
    _pad_to_4bytes(blob)
    times_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=times_offset, byteLength=len(times_bytes)))
    times_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=times_view_idx, componentType=pygltflib.FLOAT, count=num_frames, type=pygltflib.SCALAR,
        min=[float(times.min())], max=[float(times.max())]
    ))
    
    # 2. Translations Output
    trans_bytes = translations.tobytes()
    trans_offset = len(blob)
    blob.extend(trans_bytes)
    _pad_to_4bytes(blob)
    trans_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=trans_offset, byteLength=len(trans_bytes)))
    trans_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=trans_view_idx, componentType=pygltflib.FLOAT, count=num_frames, type=pygltflib.VEC3,
        min=translations.min(axis=0).tolist(), max=translations.max(axis=0).tolist()
    ))
    
    # 3. Rotations Output
    rot_bytes = rotations.tobytes()
    rot_offset = len(blob)
    blob.extend(rot_bytes)
    _pad_to_4bytes(blob)
    rot_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=rot_offset, byteLength=len(rot_bytes)))
    rot_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=rot_view_idx, componentType=pygltflib.FLOAT, count=num_frames, type=pygltflib.VEC4
    ))
    
    # Update blob length.
    gltf.set_binary_blob(blob) 
    try:
        if gltf.buffers and len(gltf.buffers) > 0:
            gltf.buffers[0].byteLength = len(blob)
    except Exception:
        pass
    
    # --- Create animation object ---
    anim = pygltflib.Animation(name="Interaction")
    
    # Translation Channel
    anim.samplers.append(pygltflib.AnimationSampler(input=times_accessor_idx, output=trans_accessor_idx, interpolation=pygltflib.LINEAR))
    anim.channels.append(pygltflib.AnimationChannel(sampler=0, target=pygltflib.AnimationChannelTarget(node=moving_node_idx, path="translation")))
    
    # Rotation Channel
    anim.samplers.append(pygltflib.AnimationSampler(input=times_accessor_idx, output=rot_accessor_idx, interpolation=pygltflib.LINEAR))
    anim.channels.append(pygltflib.AnimationChannel(sampler=1, target=pygltflib.AnimationChannelTarget(node=moving_node_idx, path="rotation")))
    
    gltf.animations.append(anim)
    
    # Save final GLB.
    gltf.save(output_path)
    
    # Cleanup.
    if os.path.exists(temp_glb_path):
        os.remove(temp_glb_path)
        
    print(f" Animated GLB saved: {output_path}")


# --- Loader ---
class FixedGAPartNetLoader(GAPartNetLoaderV2):
    """Loader for a flat directory layout (one object per subfolder)."""
    def _get_object_list(self):
        object_list = []
        if not os.path.isdir(self.dataset_root):
            print(f"Error: dataset root does not exist: '{self.dataset_root}'")
            return object_list
        for obj_id in os.listdir(self.dataset_root):
            obj_path = os.path.join(self.dataset_root, obj_id)
            if not os.path.isdir(obj_path):
                continue
            urdf_path = os.path.join(obj_path, "mobility_annotation_gapartnet.urdf")
            if os.path.exists(urdf_path):
                object_list.append(obj_path)
        return sorted(object_list)


def load_model(checkpoint_path, device, args):
    """Load the VAE model from a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error: unable to load checkpoint: {e}")
        return None, 16
    
    if 'model_state_dict' not in checkpoint:
        print("Error: checkpoint is missing 'model_state_dict'.")
        return None, 16
        
    model_state_dict = checkpoint['model_state_dict']
    config = checkpoint.get('config', {})
    
    latent_dim = config.get('latent_dim', args.latent_dim)
    num_frames = args.num_frames if args.num_frames is not None else config.get('num_frames', 16)
    transformer_layers = config.get('transformer_layers', args.transformer_layers)
    transformer_heads = config.get('transformer_heads', args.transformer_heads)
    
    print(f"\n=== Initializing VAE Model ===")
    print(f" Info: layers={transformer_layers}, heads={transformer_heads}, latent={latent_dim}")
    
    model = DualQuaternionVAE(
        latent_dim=latent_dim,
        num_frames=num_frames,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads
    ).to(device)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print(f" Loaded with strict=True.")
    except RuntimeError as e:
        print(f"Warning: Strict load failed, trying loose load. Error: {str(e)[:100]}...")
        model.load_state_dict(new_state_dict, strict=False)
            
    model.eval()
    return model, num_frames


def run_vae_diversity_test(model, loader, sample_idx, device, output_dir, num_frames, num_samples_to_gen, force_rotation=False, fps: float = 5.0, loop_mode: str = "pingpong"):
    """
    Run VAE sampling for a single input condition and export animations (GLB/GIF/MP4).
    """
    print(f"\n{'='*60}\nRunning VAE Diversity Test on sample index {sample_idx}\n{'='*60}\n")
    try:
        fps = float(fps)
    except Exception:
        fps = 5.0
    if fps <= 0:
        fps = 5.0
    if loop_mode not in ["once", "pingpong"]:
        loop_mode = "pingpong"

    sample = loader.generate_training_sample(sample_idx, num_frames=num_frames)
    if sample is None:
        print(f"Error: unable to generate a valid sample for index={sample_idx}.")
        return

    # --- 1) Prepare inputs (normalization) ---
    initial_mesh = sample['initial_mesh']
    vertex_part_mask = sample['part_mask']

    bounds = initial_mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = (bounds[1] - bounds[0]).max()
    if scale < 1e-6: scale = 1.0

    initial_mesh_normalized = initial_mesh.copy()
    initial_mesh_normalized.vertices = (initial_mesh_normalized.vertices - center) / scale
    
    initial_pc, face_indices = trimesh.sample.sample_surface(initial_mesh_normalized, 4096)
    
    sampled_part_mask_np = np.zeros(4096, dtype=np.float32)
    for i, fid in enumerate(face_indices):
        face_vertices = initial_mesh.faces[fid]
        face_mask_values = vertex_part_mask[face_vertices]
        sampled_part_mask_np[i] = np.bincount(face_mask_values.astype(int)).argmax()

    part_mask_bool_idx_torch = torch.from_numpy(vertex_part_mask).bool().to(device)

    # --- 2) Prepare ground-truth conditions ---
    drag_point_norm = (sample['drag_point'] - center) / scale
    drag_vector_norm = sample['drag_vector'] / scale
    joint_type_str = sample['joint_type']
    
    joint_type = 0 if joint_type_str in ['revolute', 'continuous'] else 1
    if force_rotation:
        joint_type = 0
        print("*** Forcing rotation mode (joint_type=0) ***")
    
    joint_axis_gt = sample['joint_axis'] 
    joint_origin_gt_norm = (sample['joint_origin'] - center) / scale

    rotation_direction_gt = sample.get('rotation_direction')
    trajectory_vectors_gt = sample.get('trajectory_vectors')
    drag_trajectory_gt = sample.get('drag_trajectory')

    # --- 3) Tensorize ---
    pc_tensor = torch.from_numpy(initial_pc).float().unsqueeze(0).to(device)
    part_mask_tensor = torch.from_numpy(sampled_part_mask_np).float().unsqueeze(0).to(device)
    drag_point_tensor = torch.from_numpy(drag_point_norm).float().unsqueeze(0).to(device)
    drag_vector_tensor = torch.from_numpy(drag_vector_norm).float().unsqueeze(0).to(device)
    joint_type_tensor = torch.tensor([joint_type], dtype=torch.long).to(device)
    joint_axis_tensor = torch.from_numpy(joint_axis_gt).float().unsqueeze(0).to(device)
    joint_origin_tensor = torch.from_numpy(joint_origin_gt_norm).float().unsqueeze(0).to(device)

    rotation_direction_tensor = None
    if rotation_direction_gt is not None:
        rotation_direction_tensor = torch.from_numpy(rotation_direction_gt).float().unsqueeze(0).to(device)
    trajectory_vectors_tensor = None
    if trajectory_vectors_gt is not None:
        trajectory_vectors_tensor = torch.from_numpy(trajectory_vectors_gt).float().unsqueeze(0).to(device)
    drag_trajectory_tensor = None

    # --- 4) VAE encoder ---
    print("Running model ENCODER...")
    with torch.no_grad():
        combined_condition_feat, joint_feat = model.encode_features(
            mesh=pc_tensor,
            part_mask=part_mask_tensor,
            drag_point=drag_point_tensor,
            drag_vector=drag_vector_tensor,
            joint_type=joint_type_tensor,
            joint_axis=joint_axis_tensor,
            joint_origin=joint_origin_tensor,
            rotation_direction=rotation_direction_tensor,
            trajectory_vectors=trajectory_vectors_tensor,
            drag_trajectory=drag_trajectory_tensor
        )
        mu = model.fc_mu(combined_condition_feat)
        logvar = model.fc_logvar(combined_condition_feat)

    # --- 5) Output directory ---
    chosen_id = os.path.basename(loader.object_list[sample_idx])
    output_dir_id = os.path.join(output_dir, chosen_id)
    os.makedirs(output_dir_id, exist_ok=True)
    
    # Export the static initial mesh as a reference.
    try:
        initial_mesh.export(os.path.join(output_dir_id, 'initial_static.glb'))
    except:
        pass

    # --- 6) VAE decoder (multi-sample) ---
    print(f"\nRunning model DECODER ({num_samples_to_gen} times)...")
    
    for s_idx in range(num_samples_to_gen):
        print(f"--- Sample {s_idx+1}/{num_samples_to_gen} ---")
        
        with torch.no_grad():
            if s_idx == 0 and num_samples_to_gen > 1:
                z = mu
            else:
                z = model.reparameterize(mu, logvar)

            pred_qr_seq, pred_qd_seq = model.decode(z, joint_feat, joint_axis_tensor, joint_type_tensor)
        
        pred_qr_seq = pred_qr_seq.squeeze(0)
        pred_qd_seq = pred_qd_seq.squeeze(0)

        # --- 7) Apply hard constraints & build a single-interaction trajectory ---
        if joint_type == 0: # rotation
            pred_qd_seq = torch.zeros_like(pred_qd_seq) 
            joint_axis_expanded = joint_axis_tensor.expand(pred_qr_seq.shape[0], -1)
            pred_qr_seq = project_rotation_to_axis(pred_qr_seq, joint_axis_expanded)
        
        elif joint_type == 1: # translation
            identity_qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            pred_qr_seq = identity_qr.unsqueeze(0).expand(pred_qr_seq.shape[0], -1)
            
            # Project translation onto the joint axis.
            pred_qr_conj = quaternion_conjugate(pred_qr_seq)
            pred_t_q = 2.0 * quaternion_multiply(pred_qd_seq, pred_qr_conj)
            pred_t_vec = pred_t_q[..., 1:]
            
            axis_vec = joint_axis_tensor.squeeze(0)
            dot_prod = torch.sum(pred_t_vec * axis_vec, dim=-1, keepdim=True)
            t_parallel = dot_prod * axis_vec
            pred_qd_seq = torch.cat([torch.zeros_like(pred_t_vec[..., 0:1]), t_parallel], dim=-1) * 0.5
        
        # --- Generate a "single interaction" trajectory (avoid cumulative spinning) ---
        t_values = _make_loop_t_values(num_frames, loop_mode=loop_mode, device=device)
        if joint_type == 0:  # Rotation: Identity -> final pose (NLERP)
            qr_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=pred_qr_seq.dtype)
            qr_final = pred_qr_seq[-1].clone()
            pred_qr_seq = _nlerp_quaternion_sequence(qr_start, qr_final, t_values)
            pred_qd_seq = torch.zeros((num_frames, 4), device=device, dtype=pred_qd_seq.dtype)
        else:  # Translation: Identity -> final translation (LERP)
            qr_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=pred_qr_seq.dtype)
            qd_start = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=pred_qd_seq.dtype)
            qr_final = qr_start
            qd_final = pred_qd_seq[-1].clone()
            # Preserve the original sign convention (translation direction may be dataset/viewer dependent).
            qd_final = -qd_final
            pred_qr_seq = qr_start.unsqueeze(0).expand(num_frames, -1)
            pred_qd_seq = (1.0 - t_values.unsqueeze(-1)) * qd_start.unsqueeze(0) + t_values.unsqueeze(-1) * qd_final.unsqueeze(0)

        # --- 8) Render video & GIF (best-effort) ---
        base_output_path = os.path.join(output_dir_id, f'predicted_z_{s_idx}')
        
        pred_mesh_sequence = []
        
        # Precompute the mesh sequence for rendering.
        joint_origin_t = joint_origin_tensor.squeeze(0)
        movable_verts = verts_normalized_torch = torch.from_numpy(initial_mesh_normalized.vertices).float().to(device)
        movable_verts_shifted = movable_verts - joint_origin_t  # shift in joint frame (mask applied later)

        for i in range(num_frames):
            qr = pred_qr_seq[i]
            qd = pred_qd_seq[i]
            
            # Apply dual-quaternion transform to the movable part only.
            # 1) gather movable vertices
            current_movable_verts = movable_verts[part_mask_bool_idx_torch]
            current_movable_verts_shifted = current_movable_verts - joint_origin_t
            
            # 2) transform
            transformed_movable = dual_quaternion_apply((qr, qd), current_movable_verts_shifted)
            transformed_movable = transformed_movable + joint_origin_t
            
            # 3) scatter back to the full mesh
            deformed_verts = movable_verts.clone()
            deformed_verts[part_mask_bool_idx_torch] = transformed_movable
            
            # 4) de-normalize to world scale
            denormalized_verts = deformed_verts.cpu().numpy() * scale + center
            
            # 5) build a mesh while preserving materials
            pred_mesh = trimesh.Trimesh(vertices=denormalized_verts, faces=initial_mesh.faces, process=False)
            pred_mesh.visual = initial_mesh.visual
            pred_mesh_sequence.append(pred_mesh)
            
        # Render video/GIF. Smaller fps -> slower playback. pingpong is more suitable for looping viewers.
        create_animation_video_local(pred_mesh_sequence, base_output_path + '.mp4', fps=fps)
        create_gif(pred_mesh_sequence, base_output_path + '.gif', fps=fps)

        # --- 9) Export a single animated GLB (no per-frame OBJ export) ---
        if pygltflib is not None:
            export_animated_glb(
                initial_mesh=initial_mesh,        # original mesh (with materials)
                part_mask_np=vertex_part_mask,    # vertex segmentation mask
                qr_seq_tensor=pred_qr_seq,        # predicted rotation sequence
                qd_seq_tensor=pred_qd_seq,        # predicted translation (DQ) sequence
                joint_origin_norm_tensor=joint_origin_tensor.squeeze(0), # normalized joint origin
                scale=scale,
                center=center,
                output_path=base_output_path + '.glb',
                fps=fps
            )

    print(f"\n{'='*60}\n VAE Diversity Test complete! Output saved to: {output_dir_id}\n{'='*60}")


# --- Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='VAE Animation Inference')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_animation_test')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--force_rotation', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_frames', type=int, default=None)
    # Playback FPS: smaller -> slower (affects MP4/GIF and GLB animation timeline).
    parser.add_argument('--fps', type=float, default=5.0, help='animation playback fps (smaller = slower)')
    parser.add_argument('--loop_mode', type=str, default='pingpong', choices=['once', 'pingpong'],
                        help='once: 0->1; pingpong: 0->1->0 (better for default looping players)')
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--transformer_heads', type=int, default=8)
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
    
    model, actual_num_frames = load_model(args.checkpoint, device, args)
    if model is None: return

    num_frames_to_gen = args.num_frames if args.num_frames is not None else actual_num_frames

    run_vae_diversity_test(
        model, loader, idx, device, args.output_dir,
        num_frames=num_frames_to_gen,
        num_samples_to_gen=args.num_samples,
        force_rotation=args.force_rotation,
        fps=args.fps,
        loop_mode=args.loop_mode
    )


if __name__ == '__main__':
    main()

def run_animation_from_sample(model, sample, sample_name, device, output_root, num_frames, num_samples_to_gen, force_rotation=False, include_groundtruth=False, fps: float = 5.0, loop_mode: str = "pingpong"):
    """Run the inference_animation pipeline on a custom sample dict."""
    initial_mesh = sample['initial_mesh']
    vertex_part_mask = np.asarray(sample['part_mask'])
    if vertex_part_mask.dtype != bool:
        vertex_part_mask = vertex_part_mask.astype(bool)
    bounds = initial_mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = (bounds[1] - bounds[0]).max()
    if scale < 1e-6:
        scale = 1.0
    initial_mesh_normalized = initial_mesh.copy()
    initial_mesh_normalized.vertices = (initial_mesh_normalized.vertices - center) / scale
    initial_pc, face_indices = trimesh.sample.sample_surface(initial_mesh_normalized, 4096)
    sampled_part_mask_np = np.zeros(4096, dtype=np.float32)
    for i, fid in enumerate(face_indices):
        face_vertices = initial_mesh.faces[fid]
        face_mask_values = vertex_part_mask[face_vertices]
        sampled_part_mask_np[i] = np.bincount(face_mask_values.astype(int)).argmax()
    part_mask_bool_idx_torch = torch.from_numpy(vertex_part_mask).bool().to(device)
    drag_point = np.asarray(sample['drag_point'])
    drag_vector = np.asarray(sample['drag_vector'])
    drag_point_norm = (drag_point - center) / scale
    drag_vector_norm = drag_vector / scale
    joint_type_raw = sample['joint_type']
    if isinstance(joint_type_raw, str):
        joint_type_str = joint_type_raw
    else:
        joint_type_str = 'revolute' if joint_type_raw == 0 else 'prismatic'
    joint_type = 0 if joint_type_str in ['revolute', 'continuous'] else 1
    if force_rotation:
        joint_type = 0
        print('*** Forcing rotation mode (joint_type=0) ***')
    joint_axis = np.asarray(sample['joint_axis'])
    joint_origin = np.asarray(sample['joint_origin'])
    joint_origin_norm = (joint_origin - center) / scale
    rotation_direction_gt = sample.get('rotation_direction')
    trajectory_vectors_gt = sample.get('trajectory_vectors')
    drag_trajectory_gt = sample.get('drag_trajectory')
    pc_tensor = torch.from_numpy(initial_pc).float().unsqueeze(0).to(device)
    part_mask_tensor = torch.from_numpy(sampled_part_mask_np).float().unsqueeze(0).to(device)
    drag_point_tensor = torch.from_numpy(drag_point_norm).float().unsqueeze(0).to(device)
    drag_vector_tensor = torch.from_numpy(drag_vector_norm).float().unsqueeze(0).to(device)
    joint_type_tensor = torch.tensor([joint_type], dtype=torch.long).to(device)
    joint_axis_tensor = torch.from_numpy(joint_axis).float().unsqueeze(0).to(device)
    joint_origin_tensor = torch.from_numpy(joint_origin_norm).float().unsqueeze(0).to(device)
    rotation_direction_tensor = None if rotation_direction_gt is None else torch.from_numpy(rotation_direction_gt).float().unsqueeze(0).to(device)
    trajectory_vectors_tensor = None if trajectory_vectors_gt is None else torch.from_numpy(trajectory_vectors_gt).float().unsqueeze(0).to(device)
    drag_trajectory_tensor = None if drag_trajectory_gt is None else torch.from_numpy(drag_trajectory_gt).float().unsqueeze(0).to(device)
    print('Running model ENCODER...')
    with torch.no_grad():
        combined_condition_feat, joint_feat = model.encode_features(
            mesh=pc_tensor,
            part_mask=part_mask_tensor,
            drag_point=drag_point_tensor,
            drag_vector=drag_vector_tensor,
            joint_type=joint_type_tensor,
            joint_axis=joint_axis_tensor,
            joint_origin=joint_origin_tensor,
            rotation_direction=rotation_direction_tensor,
            trajectory_vectors=trajectory_vectors_tensor,
            drag_trajectory=drag_trajectory_tensor
        )
        mu = model.fc_mu(combined_condition_feat)
        logvar = model.fc_logvar(combined_condition_feat)
    output_dir_id = os.path.join(output_root, sample_name)
    os.makedirs(output_dir_id, exist_ok=True)
    try:
        initial_mesh.export(os.path.join(output_dir_id, 'initial_static.glb'))
    except Exception:
        pass
    verts_normalized_torch = torch.from_numpy(initial_mesh_normalized.vertices).float().to(device)
    faces_np = initial_mesh.faces
    print(f"\nRunning model DECODER ({num_samples_to_gen} times)...")
    for s_idx in range(num_samples_to_gen):
        print(f"--- Sample {s_idx+1}/{num_samples_to_gen} ---")
        with torch.no_grad():
            if s_idx == 0 and num_samples_to_gen > 1:
                z = mu
            else:
                z = model.reparameterize(mu, logvar)
            pred_qr_seq, pred_qd_seq = model.decode(z, joint_feat, joint_axis_tensor, joint_type_tensor)
        pred_qr_seq = pred_qr_seq.squeeze(0)
        pred_qd_seq = pred_qd_seq.squeeze(0)
        if joint_type == 0:
            pred_qd_seq = torch.zeros_like(pred_qd_seq)
            axis_expanded = joint_axis_tensor.expand(pred_qr_seq.shape[0], -1)
            pred_qr_seq = project_rotation_to_axis(pred_qr_seq, axis_expanded)
        else:
            identity_qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            pred_qr_seq = identity_qr.unsqueeze(0).expand(pred_qr_seq.shape[0], -1)
            pred_qr_conj = quaternion_conjugate(pred_qr_seq)
            pred_t_q = 2.0 * quaternion_multiply(pred_qd_seq, pred_qr_conj)
            pred_t_vec = pred_t_q[..., 1:]
            axis_vec = joint_axis_tensor.squeeze(0)
            dot_prod = torch.sum(pred_t_vec * axis_vec, dim=-1, keepdim=True)
            t_parallel = dot_prod * axis_vec
            pred_qd_seq = torch.cat([torch.zeros_like(pred_t_vec[..., 0:1]), t_parallel], dim=-1) * 0.5
        # Generate a "single interaction" trajectory (avoid cumulative spinning).
        if loop_mode not in ["once", "pingpong"]:
            loop_mode = "pingpong"
        t_values = _make_loop_t_values(num_frames, loop_mode=loop_mode, device=device)
        if joint_type == 0:
            qr_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=pred_qr_seq.dtype)
            qr_final = pred_qr_seq[-1].clone()
            pred_qr_seq = _nlerp_quaternion_sequence(qr_start, qr_final, t_values)
            pred_qd_seq = torch.zeros((num_frames, 4), device=device, dtype=pred_qd_seq.dtype)
        else:
            qd_final = -pred_qd_seq[-1].clone()
            qd_start = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=pred_qd_seq.dtype)
            qr_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=pred_qr_seq.dtype)
            pred_qr_seq = qr_start.unsqueeze(0).expand(num_frames, -1)
            pred_qd_seq = (1.0 - t_values.unsqueeze(-1)) * qd_start.unsqueeze(0) + t_values.unsqueeze(-1) * qd_final.unsqueeze(0)
        base_output_path = os.path.join(output_dir_id, f'predicted_z_{s_idx}')
        pred_mesh_sequence = []
        joint_origin_t = joint_origin_tensor.squeeze(0)
        for i in range(num_frames):
            qr = pred_qr_seq[i]
            qd = pred_qd_seq[i]
            deformed_verts = verts_normalized_torch.clone()
            movable_verts = verts_normalized_torch[part_mask_bool_idx_torch]
            movable_verts_shifted = movable_verts - joint_origin_t
            transformed_movable = dual_quaternion_apply((qr, qd), movable_verts_shifted)
            transformed_movable = transformed_movable + joint_origin_t
            deformed_verts = deformed_verts.clone()
            deformed_verts[part_mask_bool_idx_torch] = transformed_movable
            denormalized_verts = deformed_verts.cpu().numpy() * scale + center
            pred_mesh = trimesh.Trimesh(vertices=denormalized_verts, faces=faces_np, process=False)
            pred_mesh.visual = initial_mesh.visual
            pred_mesh_sequence.append(pred_mesh)
        create_animation_video_local(pred_mesh_sequence, base_output_path + '.mp4', fps=fps)
        create_gif(pred_mesh_sequence, base_output_path + '.gif', fps=fps)
        if pygltflib is not None:
            export_animated_glb(
                initial_mesh=initial_mesh,
                part_mask_np=vertex_part_mask,
                qr_seq_tensor=pred_qr_seq,
                qd_seq_tensor=pred_qd_seq,
                joint_origin_norm_tensor=joint_origin_tensor.squeeze(0),
                scale=scale,
                center=center,
                output_path=base_output_path + '.glb',
                fps=fps
            )
    if include_groundtruth and 'qr_sequence' in sample and 'qd_sequence' in sample:
        print("Processing GROUND TRUTH animation (using GT KPs for comparison)...")
        gt_qr_seq = torch.from_numpy(sample['qr_sequence']).float().to(device)
        gt_qd_seq = torch.from_numpy(sample['qd_sequence']).float().to(device)
        gt_mesh_sequence = []
        gt_origin_tensor = torch.from_numpy((sample['joint_origin'] - center) / scale).float().to(device)
        for i in tqdm(range(num_frames), desc="Ground Truth"):
            qr, qd = gt_qr_seq[i], gt_qd_seq[i]
            deformed_verts = verts_normalized_torch.clone()
            movable_verts = verts_normalized_torch[part_mask_bool_idx_torch]
            joint_origin_t = gt_origin_tensor.squeeze(0)
            movable_verts_shifted = movable_verts - joint_origin_t
            transformed_movable = dual_quaternion_apply((qr, qd), movable_verts_shifted)
            transformed_movable = transformed_movable + joint_origin_t
            deformed_verts[part_mask_bool_idx_torch] = transformed_movable
            denormalized_verts = deformed_verts.cpu().numpy() * scale + center
            gt_mesh = trimesh.Trimesh(vertices=denormalized_verts, faces=faces_np, process=False)
            gt_mesh.visual = initial_mesh.visual
            gt_mesh_sequence.append(gt_mesh)
        base_gt_output_path = os.path.join(output_dir_id, 'groundtruth_gt_kps')
        create_animation_video_local(gt_mesh_sequence, base_gt_output_path + '.mp4', fps=fps)
        create_gif(gt_mesh_sequence, base_gt_output_path + '.gif', fps=fps)
        print(f"{'='*60} VAE Diversity Test complete! Output saved to: {output_dir_id}{'='*60}")
    else:
        print(f"{'='*60}VAE Animation complete! Output saved to: {output_dir_id}{'='*60}")
