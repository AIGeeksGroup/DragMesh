# -------------------------------------------------------------------
# inference_animation.py
# -------------------------------------------------------------------

import sys
import os
import torch
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
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

# --- GLTF 动画库 ---
try:
    import pygltflib
    from pygltflib import GLTF2
except ImportError:
    print("\n--- Warning: pygltflib not installed ---")
    print("Skipping animated GLB export. Install with: pip install pygltflib")
    pygltflib = None

# --- 辅助函数 ---
def project_rotation_to_axis(quaternion, target_axis):
    """(PyTorch 张量版本)"""
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

# --- 渲染函数 1: GIF ---
def create_gif(mesh_sequence, output_path, resolution=(600, 600), fps=15):
    try:
        import pyrender
        import imageio
    except ImportError:
        return

    print(f"Generating GIF: {output_path}")
    if not mesh_sequence: return
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

    renderer.delete()
    imageio.mimsave(output_path, frames, fps=fps)

# --- 渲染函数 2: MP4 ---
def create_animation_video_local(mesh_sequence, output_path, resolution=(600, 600), fps=15):
    try:
        import pyrender
        import imageio
    except ImportError:
        return

    print(f"Generating MP4: {output_path}")
    if not mesh_sequence: return
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

    renderer.delete()
    imageio.mimsave(output_path, frames, fps=fps)


# --- 核心新增函数: 导出动画 GLB ---
def export_animated_glb(initial_mesh, part_mask_np, qr_seq_tensor, qd_seq_tensor, 
                        joint_origin_norm_tensor, scale, center, output_path, fps=10):
    """
    生成包含动画轨道的 GLB 文件。
    """
    if pygltflib is None:
        return

    print(f"Generating Animated GLB: {output_path}")

    # 1. 准备数据: 还原真实尺度
    # VAE输出是归一化的，我们需要还原到原始尺寸
    real_origin = joint_origin_norm_tensor.cpu().numpy() * scale + center
    
    # 2. 拆分网格 (Static vs Moving)
    # 使用 trimesh 的 submesh 功能保留贴图
    try:
        # 获取所有面
        faces = initial_mesh.faces
        vertex_mask = part_mask_np.astype(bool)
        
        # 判定面的归属: 如果一个面的所有顶点都是可动点，则归为 moving，否则归为 static
        # (这是一种简单的策略，对于 GAPartNet 这种刚性分割通常有效)
        face_mask = vertex_mask[faces].all(axis=1)
        
        # 创建子网格
        mesh_moving = initial_mesh.submesh([face_mask], append=True)
        mesh_static = initial_mesh.submesh([~face_mask], append=True)
        
        # 命名，方便调试
        mesh_moving.name = "moving_part"
        mesh_static.name = "static_part"
        
    except Exception as e:
        print(f"Error splitting mesh for GLB: {e}")
        # 如果拆分失败，尝试导出整个静态物体作为 fallback
        initial_mesh.export(output_path)
        return

    # 3. 设置节点坐标系 (Pivot Adjustment)
    # 关键步骤：为了让 GLTF 动画绕着关节轴旋转，我们需要：
    # A. 将 Moving Mesh 的顶点移动到以 (0,0,0) 为关节中心的位置 ( Vertex - Origin )
    # B. 将承载该 Mesh 的 Node 移动回关节在世界坐标的位置 ( Node Position = Origin )
    
    mesh_moving.vertices -= real_origin
    
    # 4. 构建 Trimesh Scene 并导出基础 GLB
    scene = trimesh.Scene()
    scene.add_geometry(mesh_static, node_name='static_node')
    
    # 初始变换矩阵: 平移到 pivot 点
    transform_matrix = np.eye(4)
    transform_matrix[:3, 3] = real_origin
    scene.add_geometry(mesh_moving, node_name='moving_node', transform=transform_matrix)
    
    # 导出临时 GLB (利用 trimesh 处理贴图打包和二进制转换)
    fd, temp_glb_path = tempfile.mkstemp(suffix='.glb')
    os.close(fd)
    scene.export(temp_glb_path)
    
    # 5. 使用 pygltflib 注入动画数据
    gltf = GLTF2().load(temp_glb_path)
    
    # 找到 moving_node 的索引
    moving_node_idx = -1
    for i, node in enumerate(gltf.nodes):
        if node.name == 'moving_node':
            moving_node_idx = i
            break
    
    # 如果找不到名字(有时候trimesh会改名)，做个兜底猜测: 最后一个 node 通常是最后加的
    if moving_node_idx == -1:
         moving_node_idx = len(gltf.nodes) - 1

    # --- 准备动画关键帧数据 ---
    num_frames = qr_seq_tensor.shape[0]
    times = np.linspace(0, num_frames / fps, num_frames, dtype=np.float32)
    
    # A. 旋转转换 (DualQuaternion Real Part -> GLTF Quaternion)
    # PyTorch: [w, x, y, z] -> GLTF: [x, y, z, w]
    qr_seq = qr_seq_tensor.cpu().numpy()
    rotations = np.zeros((num_frames, 4), dtype=np.float32)
    rotations[:, 0] = qr_seq[:, 1] # x
    rotations[:, 1] = qr_seq[:, 2] # y
    rotations[:, 2] = qr_seq[:, 3] # z
    rotations[:, 3] = qr_seq[:, 0] # w
    
    # B. 平移转换
    # 我们的 Node 已经在 real_origin 了。
    # VAE 预测的平移 t (from DQ) 是相对于原点的偏移。
    # 所以每一帧 Node 的位置 = real_origin + t * scale
    
    qr_conj = quaternion_conjugate(qr_seq_tensor)
    t_q = 2.0 * quaternion_multiply(qd_seq_tensor, qr_conj)
    t_vec_norm = t_q[:, 1:].cpu().numpy() # [N, 3]
    t_vec_real = t_vec_norm * scale 
    
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    for i in range(num_frames):
        translations[i] = real_origin + t_vec_real[i]

    # --- 写入 Binary Buffer ---
    blob = bytearray(gltf.binary_blob())
    
    # 1. Times Input
    times_bytes = times.tobytes()
    times_offset = len(blob)
    blob.extend(times_bytes)
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
    rot_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=rot_offset, byteLength=len(rot_bytes)))
    rot_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=rot_view_idx, componentType=pygltflib.FLOAT, count=num_frames, type=pygltflib.VEC4
    ))
    
    # 更新 blob
    gltf.set_binary_blob(blob) 
    
    # --- 创建 Animation 对象 ---
    anim = pygltflib.Animation(name="Interaction")
    
    # Translation Channel
    anim.samplers.append(pygltflib.AnimationSampler(input=times_accessor_idx, output=trans_accessor_idx, interpolation=pygltflib.LINEAR))
    anim.channels.append(pygltflib.AnimationChannel(sampler=0, target=pygltflib.AnimationChannelTarget(node=moving_node_idx, path="translation")))
    
    # Rotation Channel
    anim.samplers.append(pygltflib.AnimationSampler(input=times_accessor_idx, output=rot_accessor_idx, interpolation=pygltflib.LINEAR))
    anim.channels.append(pygltflib.AnimationChannel(sampler=1, target=pygltflib.AnimationChannelTarget(node=moving_node_idx, path="rotation")))
    
    gltf.animations.append(anim)
    
    # 保存最终文件
    gltf.save(output_path)
    
    # 清理临时文件
    if os.path.exists(temp_glb_path):
        os.remove(temp_glb_path)
        
    print(f" Animated GLB saved: {output_path}")


# --- Loader ---
class FixedGAPartNetLoader(GAPartNetLoaderV2):
    """适配平铺目录结构"""
    def _get_object_list(self):
        object_list = []
        if not os.path.isdir(self.dataset_root):
            print(f"Error: 数据集根目录不存在: '{self.dataset_root}'")
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
    """从 checkpoint 加载 VAE 模型。"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error: 无法加载 checkpoint: {e}")
        return None, 16
    
    if 'model_state_dict' not in checkpoint:
        print(f"❌ 错误: Checkpoint 中没有找到 'model_state_dict'。")
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


def run_vae_diversity_test(model, loader, sample_idx, device, output_dir, num_frames, num_samples_to_gen, force_rotation=False):
    """
    运行 VAE 多样性测试 (已添加 GLB 导出，移除逐帧 OBJ)
    """
    print(f"\n{'='*60}\nRunning VAE Diversity Test on sample index {sample_idx}\n{'='*60}\n")

    sample = loader.generate_training_sample(sample_idx, num_frames=num_frames)
    if sample is None:
        print(f"Error: 无法为 index={sample_idx} 生成有效样本。")
        return

    # --- 1. 数据准备 (归一化) ---
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

    # --- 2. 准备所有 *真值* (GT) 条件 ---
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

    # --- 3. 张量化 ---
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

    # --- 4. 运行 VAE Encoder ---
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

    # --- 5. 准备输出目录 ---
    chosen_id = os.path.basename(loader.object_list[sample_idx])
    output_dir_id = os.path.join(output_dir, chosen_id)
    os.makedirs(output_dir_id, exist_ok=True)
    
    # 导出一次初始 GLB 作为参考
    try:
        initial_mesh.export(os.path.join(output_dir_id, 'initial_static.glb'))
    except:
        pass

    # --- 6. 运行 VAE Decoder (多次采样) ---
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

        # --- 7. 应用硬约束 & 累积/插值逻辑 ---
        if joint_type == 0: # 旋转
            pred_qd_seq = torch.zeros_like(pred_qd_seq) 
            joint_axis_expanded = joint_axis_tensor.expand(pred_qr_seq.shape[0], -1)
            pred_qr_seq = project_rotation_to_axis(pred_qr_seq, joint_axis_expanded)
        
        elif joint_type == 1: # 平移
            identity_qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            pred_qr_seq = identity_qr.unsqueeze(0).expand(pred_qr_seq.shape[0], -1)
            
            # 平移投影
            pred_qr_conj = quaternion_conjugate(pred_qr_seq)
            pred_t_q = 2.0 * quaternion_multiply(pred_qd_seq, pred_qr_conj)
            pred_t_vec = pred_t_q[..., 1:]
            
            axis_vec = joint_axis_tensor.squeeze(0)
            dot_prod = torch.sum(pred_t_vec * axis_vec, dim=-1, keepdim=True)
            t_parallel = dot_prod * axis_vec
            pred_qd_seq = torch.cat([torch.zeros_like(pred_t_vec[..., 0:1]), t_parallel], dim=-1) * 0.5
        
        # --- 分支逻辑: 累积 vs 插值 ---
        if joint_type == 0: # 旋转: 累积
            # 强制第一帧为 Identity
            pred_qr_seq[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            pred_qd_seq[0] = torch.zeros(4, device=device)
            
            final_pred_qr_seq = torch.empty_like(pred_qr_seq)
            final_pred_qd_seq = torch.empty_like(pred_qd_seq)
            cumulative_qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            cumulative_qd = torch.zeros(4, device=device)

            for i in range(num_frames):
                cumulative_qr = quaternion_multiply(pred_qr_seq[i], cumulative_qr) 
                cumulative_qd = quaternion_multiply(pred_qd_seq[i], cumulative_qd) 
                final_pred_qr_seq[i] = cumulative_qr
                final_pred_qd_seq[i] = cumulative_qd
            
            pred_qr_seq = final_pred_qr_seq
            pred_qd_seq = final_pred_qd_seq

        else: # 平移: 插值 (LERP)
            qr_final = pred_qr_seq[-1].clone()
            qd_final = pred_qd_seq[-1].clone()
            
            # 强制反转 (原代码逻辑)
            qd_final = -qd_final
            
            qr_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=qr_final.dtype)
            qd_start = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=qd_final.dtype)
            t_values = torch.linspace(0.0, 1.0, steps=num_frames, device=device)
            
            # Rotation Interpolation
            dot = torch.dot(qr_start, qr_final)
            if dot < 0.0: qr_final = -qr_final 
            
            qr_interpolated = (1.0 - t_values.unsqueeze(-1)) * qr_start.unsqueeze(0) + \
                              t_values.unsqueeze(-1) * qr_final.unsqueeze(0)
            pred_qr_seq = F.normalize(qr_interpolated, p=2, dim=-1)

            # Translation Interpolation
            pred_qd_seq = (1.0 - t_values.unsqueeze(-1)) * qd_start.unsqueeze(0) + \
                          t_values.unsqueeze(-1) * qd_final.unsqueeze(0)

        # --- 8. 生成视频 & GIF ---
        base_output_path = os.path.join(output_dir_id, f'predicted_z_{s_idx}')
        
        pred_mesh_sequence = []
        
        # 预计算用于视频渲染的 sequence
        joint_origin_t = joint_origin_tensor.squeeze(0)
        movable_verts = verts_normalized_torch = torch.from_numpy(initial_mesh_normalized.vertices).float().to(device)
        movable_verts_shifted = movable_verts - joint_origin_t # 全体shift，之后mask选部分

        for i in range(num_frames):
            qr = pred_qr_seq[i]
            qd = pred_qd_seq[i]
            
            # 只对 movable part 进行 dual quaternion 变换
            # 1. 取出 movable verts
            current_movable_verts = movable_verts[part_mask_bool_idx_torch]
            current_movable_verts_shifted = current_movable_verts - joint_origin_t
            
            # 2. 变换
            transformed_movable = dual_quaternion_apply((qr, qd), current_movable_verts_shifted)
            transformed_movable = transformed_movable + joint_origin_t
            
            # 3. 拼回完整 mesh
            deformed_verts = movable_verts.clone()
            deformed_verts[part_mask_bool_idx_torch] = transformed_movable
            
            # 4. 还原尺度
            denormalized_verts = deformed_verts.cpu().numpy() * scale + center
            
            # 5. 创建带贴图的 mesh
            pred_mesh = trimesh.Trimesh(vertices=denormalized_verts, faces=initial_mesh.faces, process=False)
            pred_mesh.visual = initial_mesh.visual
            pred_mesh_sequence.append(pred_mesh)
            
        # 渲染视频和GIF
        create_animation_video_local(pred_mesh_sequence, base_output_path + '.mp4', fps=10)
        create_gif(pred_mesh_sequence, base_output_path + '.gif', fps=10)

        # --- 9. 【关键】生成单个交互式 GLB 文件 ---
        # 只要这一步，不再生成逐帧 obj
        if pygltflib is not None:
            export_animated_glb(
                initial_mesh=initial_mesh,        # 原始带贴图网格
                part_mask_np=vertex_part_mask,    # 分割掩码
                qr_seq_tensor=pred_qr_seq,        # 预测的旋转序列
                qd_seq_tensor=pred_qd_seq,        # 预测的平移(DQ)序列
                joint_origin_norm_tensor=joint_origin_tensor.squeeze(0), # 归一化的关节中心
                scale=scale,
                center=center,
                output_path=base_output_path + '.glb',
                fps=10
            )

    print(f"\n{'='*60}\n VAE Diversity Test complete! Output saved to: {output_dir_id}\n{'='*60}")


# --- (参数) ---
def parse_args():
    parser = argparse.ArgumentParser(description='VAE Animation Inference')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_animation_test')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--force_rotation', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--transformer_layers', type=int, default=4)
    parser.add_argument('--transformer_heads', type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loader = FixedGAPartNetLoader(dataset_root=args.dataset_root)
    if len(loader.object_list) == 0:
        print("Error: 数据集为空。")
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
        force_rotation=args.force_rotation
    )


if __name__ == '__main__':
    main()

def run_animation_from_sample(model, sample, sample_name, device, output_root, num_frames, num_samples_to_gen, force_rotation=False, include_groundtruth=False):
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
        if joint_type == 0:
            pred_qr_seq[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            pred_qd_seq[0] = torch.zeros(4, device=device)
            final_qr = torch.empty_like(pred_qr_seq)
            final_qd = torch.empty_like(pred_qd_seq)
            cumulative_qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            cumulative_qd = torch.zeros(4, device=device)
            for i in range(num_frames):
                cumulative_qr = quaternion_multiply(pred_qr_seq[i], cumulative_qr)
                cumulative_qd = quaternion_multiply(pred_qd_seq[i], cumulative_qd)
                final_qr[i] = cumulative_qr
                final_qd[i] = cumulative_qd
            pred_qr_seq = final_qr
            pred_qd_seq = final_qd
        else:
            qr_final = pred_qr_seq[-1].clone()
            qd_final = pred_qd_seq[-1].clone()
            qd_final = -qd_final
            qr_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=qr_final.dtype)
            qd_start = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=qd_final.dtype)
            t_values = torch.linspace(0.0, 1.0, steps=num_frames, device=device)
            dot = torch.dot(qr_start, qr_final)
            if dot < 0.0:
                qr_final = -qr_final
            qr_interp = (1.0 - t_values.unsqueeze(-1)) * qr_start.unsqueeze(0) + t_values.unsqueeze(-1) * qr_final.unsqueeze(0)
            pred_qr_seq = F.normalize(qr_interp, p=2, dim=-1)
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
        create_animation_video_local(pred_mesh_sequence, base_output_path + '.mp4', fps=10)
        create_gif(pred_mesh_sequence, base_output_path + '.gif', fps=10)
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
                fps=10
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
        create_animation_video_local(gt_mesh_sequence, base_gt_output_path + '.mp4', fps=10)
        create_gif(gt_mesh_sequence, base_gt_output_path + '.gif', fps=10)
        print(f"{'='*60} VAE Diversity Test complete! Output saved to: {output_dir_id}{'='*60}")
    else:
        print(f"{'='*60}VAE Animation complete! Output saved to: {output_dir_id}{'='*60}")
