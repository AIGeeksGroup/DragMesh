# -------------------------------------------------------------------
#  inference_pipeline.py
# -------------------------------------------------------------------

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.model_v2 import DualQuaternionVAE
from modules.predictor import KeypointPredictor
from modules.dual_quaternion import quaternion_conjugate, quaternion_multiply
from inference_animation import run_animation_from_sample

try:
    import requests
except ImportError:
    requests = None

try:
    import viser
except ImportError:
    viser = None

try:
    from trimesh.ray.ray_triangle import RayMeshIntersector
except ImportError:
    RayMeshIntersector = None



def face_to_vertex_labels(faces: np.ndarray, num_vertices: int, face_labels: np.ndarray) -> np.ndarray:
    vertex_faces: List[List[int]] = [[] for _ in range(num_vertices)]
    for fid, face in enumerate(faces):
        label = int(face_labels[fid])
        if label < 0:
            continue
        for vid in face:
            vertex_faces[vid].append(label)
    vertex_labels = -np.ones(num_vertices, dtype=np.int64)
    for vid, labels in enumerate(vertex_faces):
        if not labels:
            continue
        counts = np.bincount(np.array(labels, dtype=np.int64))
        vertex_labels[vid] = int(np.argmax(counts))
    return vertex_labels


def create_bool_mask(indices: np.ndarray, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    mask[indices] = True
    return mask


def estimate_drag_handle(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    shifted = vertices - center
    cov = np.cov(shifted.T) if vertices.shape[0] >= 3 else np.eye(3)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    axis = eig_vecs[:, np.argmax(eig_vals)]
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-6:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis = axis / norm_axis
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    magnitude = max(float(np.max(extents)) * 0.5, 0.05)
    return center, axis * magnitude


class SegmentedPart:
    def __init__(self, part_id: int, vertex_mask: np.ndarray, drag_point: np.ndarray,
                 drag_vector: np.ndarray, num_vertices: int):
        self.part_id = part_id
        self.vertex_mask = vertex_mask
        self.drag_point = drag_point
        self.drag_vector = drag_vector
        self.num_vertices = num_vertices
        self.drag_point_source = "heuristic"


def create_segmented_part(mesh: trimesh.Trimesh,
                          vertex_labels: np.ndarray,
                          label: int) -> Optional[SegmentedPart]:
    indices = np.where(vertex_labels == label)[0]
    if indices.size == 0:
        return None
    verts = mesh.vertices[indices]
    drag_point, drag_vector = estimate_drag_handle(verts)
    mask = create_bool_mask(indices, len(vertex_labels))
    return SegmentedPart(label, mask, drag_point, drag_vector, indices.size)




def launch_drag_picker(mesh: trimesh.Trimesh,
                       vertex_labels: np.ndarray,
                       target_label: Optional[int],
                       drag_point_default: Optional[np.ndarray],
                       drag_vector_default: Optional[np.ndarray],
                       args) -> Tuple[int, np.ndarray, np.ndarray]:
    if viser is None:
        raise RuntimeError("Interactive mode requires 'viser'. Install via `pip install viser`.")

    server = viser.ViserServer(host=args.interactive_host, port=args.interactive_port)
    server.scene.set_up_direction("+y")

    view_mesh = mesh.copy()
    bounds = view_mesh.bounds
    center_view = (bounds[0] + bounds[1]) / 2.0
    scale_view = (bounds[1] - bounds[0]).max()
    if scale_view < 1e-6:
        scale_view = 1.0
    view_mesh.vertices = (view_mesh.vertices - center_view) / scale_view

    camera_position = np.array([1.5, 1.5, 1.5])
    camera_look_at = np.zeros(3)
    if hasattr(server.scene, "camera"):
        server.scene.camera.position = camera_position
        server.scene.camera.look_at = camera_look_at
    elif hasattr(server.scene, "set_camera_default"):
        server.scene.set_camera_default(position=camera_position,
                                        look_at=camera_look_at,
                                        up=np.array([0.0, 1.0, 0.0]))

    def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return r, g, b

    points_view = view_mesh.vertices
    points_world = mesh.vertices
    point_size = max(np.linalg.norm(mesh.extents) * 0.003, 0.002)
    color_map: Dict[int, np.ndarray] = {}
    unique_labels = [int(v) for v in np.unique(vertex_labels) if v >= 0]
    for label in unique_labels:
        hue = (0.61803398875 * (label + 1)) % 1.0
        rgb = hsv_to_rgb(hue, 0.65, 0.95)
        color_map[label] = (np.array(rgb) * 255).astype(np.uint8)
    face_vertex_labels = vertex_labels[mesh.faces]
    covered_faces = np.zeros(len(mesh.faces), dtype=bool)
    part_mesh_cache: Dict[int, trimesh.Trimesh] = {}

    for label in unique_labels:
        face_ids = np.where(np.all(face_vertex_labels == label, axis=1))[0]
        if face_ids.size == 0:
            continue
        part_mesh = view_mesh.submesh([face_ids], append=True)
        part_mesh_cache[label] = part_mesh
        covered_faces[face_ids] = True
        server.scene.add_mesh_simple(
            name=f"/part_mesh_{label}",
            vertices=part_mesh.vertices,
            faces=part_mesh.faces,
            color=tuple(int(c) for c in color_map[label])
        )

    remaining = np.where(~covered_faces)[0]
    if remaining.size > 0:
        remaining_mesh = view_mesh.submesh([remaining], append=True)
        server.scene.add_mesh_simple(
            name="/part_mesh_unlabeled",
            vertices=remaining_mesh.vertices,
            faces=remaining_mesh.faces,
            color=(180, 180, 180)
        )

    if args.interactive_show_points and unique_labels:
        point_colors = np.array([color_map.get(int(label), np.array([200, 200, 200], dtype=np.uint8))
                                 for label in vertex_labels])
        server.scene.add_point_cloud(
            name="/point_cloud_overlay",
            points=points_view,
            colors=point_colors,
            point_size=point_size
        )

    ray_intersector = None
    if RayMeshIntersector is not None:
        try:
            ray_intersector = RayMeshIntersector(view_mesh)
        except Exception:
            ray_intersector = None

    state = {
        "part_id": target_label,
        "point": drag_point_default,
        "vector": drag_vector_default,
        "confirmed": False
    }

    marker_point_name = "/drag_point_marker"
    marker_vector_name = "/drag_vector_marker"
    marker_vector_end_name = "/drag_vector_end"
    highlight_name = "/part_highlight"

    def to_view_coords(world_point: np.ndarray) -> np.ndarray:
        return (world_point - center_view) / scale_view

    def remove_node(name: str):
        if hasattr(server.scene, "remove_node"):
            try:
                server.scene.remove_node(name)
            except KeyError:
                pass

    def refresh_highlight():
        remove_node(highlight_name)
        if state["part_id"] is None:
            return
        part_mesh = part_mesh_cache.get(state["part_id"])
        if part_mesh is None:
            return
        server.scene.add_mesh_simple(
            name=highlight_name,
            vertices=part_mesh.vertices,
            faces=part_mesh.faces,
            color=(255, 230, 80)
        )

    def refresh_markers():
        remove_node(marker_point_name)
        remove_node(marker_vector_name)
        remove_node(marker_vector_end_name)
        if state["point"] is not None:
            point_view = to_view_coords(state["point"])
            if hasattr(server.scene, "add_sphere"):
                server.scene.add_sphere(
                    name=marker_point_name,
                    center=point_view,
                    radius=max(0.01, point_size * 2.0),
                    color=(255, 90, 90)
                )
            else:
                server.scene.add_point_cloud(
                    name=marker_point_name,
                    points=point_view[None, :],
                    colors=np.array([[255, 90, 90]], dtype=np.uint8),
                    point_size=point_size * 3.0
                )
        if state["point"] is not None and state["vector"] is not None:
            start = to_view_coords(state["point"])
            end = to_view_coords(state["point"] + state["vector"])
            if hasattr(server.scene, "add_arrow"):
                server.scene.add_arrow(
                    name=marker_vector_name,
                    start=start,
                    end=end,
                    color=(80, 180, 255),
                    radius=max(0.004, point_size)
                )
            else:
                line_points = np.vstack([start, end])
                server.scene.add_point_cloud(
                    name=marker_vector_name,
                    points=line_points,
                    colors=np.array([[80, 180, 255],
                                     [80, 180, 255]], dtype=np.uint8),
                    point_size=point_size * 2.0
                )
            if hasattr(server.scene, "add_sphere"):
                server.scene.add_sphere(
                    name=marker_vector_end_name,
                    center=end,
                    radius=max(0.01, point_size * 2.0),
                    color=(120, 255, 120)
                )
            else:
                server.scene.add_point_cloud(
                    name=marker_vector_end_name,
                    points=end[None, :],
                    colors=np.array([[120, 255, 120]], dtype=np.uint8),
                    point_size=point_size * 3.0
                )

    refresh_markers()
    refresh_highlight()

    def fallback_nearest(ray_origin: np.ndarray, ray_dir: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        diff = points_view - ray_origin
        t = diff @ ray_dir
        t = np.clip(t, 0.0, np.linalg.norm(points_view.ptp(axis=0)))
        closest = ray_origin + np.outer(t, ray_dir)
        dists = np.linalg.norm(points_view - closest, axis=1)
        idx = int(np.argmin(dists))
        label = int(vertex_labels[idx]) if idx is not None else -1
        return label, points_world[idx], points_view[idx]

    def fallback_ray_target(ray_origin: np.ndarray, ray_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        default_dist = max(float(scale_view), 0.5)
        world_point = ray_origin + ray_dir * default_dist
        return world_point, to_view_coords(world_point)

    def pick_location(ray_origin: np.ndarray,
                      ray_dir: np.ndarray,
                      require_hit: bool) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
        if ray_intersector is not None:
            try:
                locations, _, tri_ids = ray_intersector.intersects_location(
                    np.array([ray_origin], dtype=np.float64),
                    np.array([ray_dir], dtype=np.float64)
                )
            except Exception:
                locations = np.zeros((0, 3))
                tri_ids = np.zeros((0,), dtype=np.int64)
            if locations.shape[0] > 0:
                loc_view = locations[0]
                tri_id = int(tri_ids[0])
                tri_labels = vertex_labels[mesh.faces[tri_id]]
                valid = tri_labels[tri_labels >= 0]
                if valid.size > 0:
                    counts = np.bincount(valid)
                    label = int(np.argmax(counts))
                else:
                    label = int(tri_labels[0])
                world_point = loc_view * scale_view + center_view
                return label, world_point, loc_view
        if require_hit:
            label, world_point, view_point = fallback_nearest(ray_origin, ray_dir)
            return label, world_point, view_point
        world_point, view_point = fallback_ray_target(ray_origin, ray_dir)
        return None, world_point, view_point

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        status = client.gui.add_markdown("Select part / drag point / direction.")

        valid_labels = vertex_labels[vertex_labels >= 0]
        slider_min = int(valid_labels.min()) if valid_labels.size > 0 else 0
        slider_max = int(valid_labels.max()) if valid_labels.size > 0 else 0
        initial_val = int(state["part_id"]) if state["part_id"] is not None else slider_min
        use_int_slider = hasattr(client.gui, "add_slider_int")
        if use_int_slider:
            part_slider = client.gui.add_slider_int(
                "Part ID",
                min=slider_min,
                max=slider_max,
                step=1,
                initial_value=initial_val
            )
        else:
            part_slider = client.gui.add_slider(
                "Part ID",
                min=slider_min,
                max=slider_max,
                step=1,
                initial_value=float(initial_val)
            )

        point_btn = client.gui.add_button("Pick Drag Point", icon=viser.Icon.POINTER)
        vector_btn = client.gui.add_button("Pick Drag Direction", icon=viser.Icon.ARROW_RIGHT)
        confirm_btn = client.gui.add_button("Confirm", icon=viser.Icon.CHECK)
        cancel_btn = client.gui.add_button("Cancel", icon=viser.Icon.X)

        def set_status(msg: str):
            status.content = msg

        @part_slider.on_update
        def _(_value):
            slider_value = part_slider.value
            if not use_int_slider:
                slider_value = int(round(slider_value))
            state["part_id"] = slider_value
            set_status(f"Target part set to {state['part_id']}.")
            refresh_highlight()

        def pick(mode: str):
            button = point_btn if mode == "point" else vector_btn
            button.disabled = True

            @client.scene.on_pointer_event(event_type="click")
            def _(event: viser.ScenePointerEvent) -> None:
                ray_origin = np.array(event.ray_origin, dtype=np.float64)
                ray_dir = np.array(event.ray_direction, dtype=np.float64)
                label, world_point, view_point = pick_location(
                    ray_origin,
                    ray_dir,
                    require_hit=(mode == "point")
                )
                if mode == "point":
                    if label is None or label < 0:
                        set_status("Click on a labeled part to pick the drag point.")
                        client.scene.remove_pointer_callback()
                        return
                    state["part_id"] = label
                    state["point"] = world_point
                    state["vector"] = None
                    set_status(f"Drag point set. Part {label}.")
                    refresh_highlight()
                else:
                    if state["point"] is None:
                        set_status("Pick drag point first.")
                        return
                    state["vector"] = world_point - state["point"]
                    set_status(f"Drag vector set (length {np.linalg.norm(state['vector']):.3f}).")
                refresh_markers()
                client.scene.remove_pointer_callback()

            @client.scene.on_pointer_callback_removed
            def _():
                button.disabled = False

        @point_btn.on_click
        def _(_):
            pick("point")

        @vector_btn.on_click
        def _(_):
            pick("vector")

        @confirm_btn.on_click
        def _(_):
            if state["part_id"] is None or state["point"] is None or state["vector"] is None:
                set_status("Part / point / direction must be set before confirming.")
                return
            state["confirmed"] = True
            if hasattr(server, "stop"):
                server.stop()

        @cancel_btn.on_click
        def _(_):
            state["confirmed"] = False
            state["part_id"] = None
            if hasattr(server, "stop"):
                server.stop()

    while True:
        if hasattr(server, "is_running"):
            if not server.is_running:
                break
        else:
            if state["confirmed"] or state["part_id"] is None:
                break
        time.sleep(0.1)

    if not state["confirmed"] or state["part_id"] is None or state["point"] is None or state["vector"] is None:
        raise RuntimeError("Drag selection incomplete.")
    return state["part_id"], state["point"], state["vector"]



def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    normalized = mesh.copy()
    bounds = normalized.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = (bounds[1] - bounds[0]).max()
    if scale < 1e-6:
        scale = 1.0
    normalized.vertices = (normalized.vertices - center) / scale
    pc, face_indices = trimesh.sample.sample_surface(normalized, num_points)
    return pc, face_indices, center, scale


def run_kpp_inference(kpp_model: KeypointPredictor,
                      mesh_pc: np.ndarray,
                      sampled_mask: np.ndarray,
                      drag_point_norm: np.ndarray,
                      drag_vector_norm: np.ndarray,
                      device: torch.device):
    pc_tensor = torch.from_numpy(mesh_pc).float().unsqueeze(0).to(device)
    part_mask_tensor = torch.from_numpy(sampled_mask).float().unsqueeze(0).to(device)
    drag_point_tensor = torch.from_numpy(drag_point_norm).float().unsqueeze(0).to(device)
    drag_vector_tensor = torch.from_numpy(drag_vector_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_type_logits, pred_axis, pred_origin = kpp_model(
            pc_tensor,
            part_mask_tensor,
            drag_point_tensor,
            drag_vector_tensor
        )
    return pred_type_logits, pred_axis, pred_origin


def load_kpp_model(checkpoint_path: str, device: torch.device) -> KeypointPredictor:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    kpp_model = KeypointPredictor(
        use_mask=config.get('use_mask', True),
        use_drag=config.get('use_drag', True)
    ).to(device)
    state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    kpp_model.load_state_dict(state, strict=True)
    kpp_model.eval()
    return kpp_model


def load_vae_model(checkpoint_path: str, device: torch.device, override_frames: Optional[int]) -> Tuple[DualQuaternionVAE, int]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 256)
    num_frames = override_frames if override_frames is not None else config.get('num_frames', 16)
    vae_model = DualQuaternionVAE(
        latent_dim=latent_dim,
        num_frames=num_frames,
        transformer_layers=config.get('transformer_layers', 4),
        transformer_heads=config.get('transformer_heads', 8)
    ).to(device)
    state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    vae_model.load_state_dict(state, strict=True)
    vae_model.eval()
    return vae_model, num_frames



class LLMJointClassifier:
    def __init__(self, endpoint: Optional[str], api_key: Optional[str],
                 model: str, temperature: float, timeout: float,
                 system_prompt: str, max_retries: int, proxy: Optional[str]):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.max_retries = max(1, max_retries)
        self.proxy = {"http": proxy, "https": proxy} if proxy else None
        self.enabled = bool(endpoint and requests is not None)
        self.session = requests.Session() if self.enabled else None
        if endpoint and requests is None:
            print("requests not installed; LLM disabled.")

    def classify(self, prompt: str) -> Tuple[Optional[int], Optional[str]]:
        if not self.enabled or self.session is None:
            return None, None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 16
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    proxies=self.proxy
                )
                response.raise_for_status()
                content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                norm = content.strip().lower()
                if any(word in norm for word in ['revolute', 'rotate', 'hinge']):
                    return 0, content
                if any(word in norm for word in ['prismatic', 'slide', 'linear']):
                    return 1, content
                return None, content
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        if last_error:
            print(f"LLM request failed: {last_error}")
        return None, None



def parse_vec3(value: str) -> np.ndarray:
    parts = [float(v) for v in value.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 values, got '{value}'")
    return np.array(parts, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive inference pipeline (mesh + drag UI).")
    parser.add_argument('--mesh_file', type=str, required=True,
                        help='Path to OBJ/GLB mesh (already segmented).')
    parser.add_argument('--mask_file', type=str, default=None,
                        help='Optional .npy mask (face ids or vertex labels).')
    parser.add_argument('--mask_format', choices=['face', 'vertex'], default='face',
                        help='Interpretation of mask_file.')
    parser.add_argument('--part_id', type=int, default=None,
                        help='Initial part ID to target (optional).')

    parser.add_argument('--output_dir', type=str, default='results_manual')
    parser.add_argument('--kpp_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)

    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--num_frames', type=int, default=None)

    parser.add_argument('--drag_point', type=parse_vec3, default=None,
                        help='Manual drag point (x,y,z).')
    parser.add_argument('--drag_vector', type=parse_vec3, default=None,
                        help='Manual drag vector (x,y,z).')
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive viewer to pick drag point/direction.')
    parser.add_argument('--interactive_host', type=str, default='127.0.0.1')
    parser.add_argument('--interactive_port', type=int, default=8123)
    parser.add_argument('--interactive_max_points', type=int, default=60000,
                        help='Max points rendered in interactive viewer')
    parser.add_argument('--interactive_show_points', action='store_true',
                        help='Render point cloud overlay in interactive viewer (default: mesh only)')

    parser.add_argument('--manual_joint_type', choices=['revolute', 'prismatic'], default=None,
                        help='Override joint type manually.')
    parser.add_argument('--llm_endpoint', type=str, default=None)
    parser.add_argument('--llm_api_key', type=str, default=None)
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--llm_temperature', type=float, default=0.0)
    parser.add_argument('--llm_timeout', type=float, default=8.0)
    parser.add_argument('--llm_max_retries', type=int, default=1)
    parser.add_argument('--llm_proxy', type=str, default=None)
    parser.add_argument('--llm_system_prompt', type=str,
                        default="You are an articulation expert. Reply with 'revolute' or 'prismatic'.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.drag_vector is not None and args.drag_point is None and not args.interactive:
        raise ValueError("Provide --drag_point when specifying --drag_vector.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    mesh = trimesh.load(args.mesh_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise RuntimeError(f"Scene '{args.mesh_file}' contains no geometry.")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    print(f"Loaded mesh: {args.mesh_file} ({len(mesh.vertices)} verts)")

    if args.mask_file is not None:
        mask_array = np.load(args.mask_file)
        if args.mask_format == 'face':
            if mask_array.shape[0] != len(mesh.faces):
                raise ValueError("face mask length mismatch.")
            vertex_labels = face_to_vertex_labels(mesh.faces, len(mesh.vertices), mask_array)
        else:
            if mask_array.shape[0] != len(mesh.vertices):
                raise ValueError("vertex mask length mismatch.")
            vertex_labels = mask_array.astype(int)
    else:
        vertex_labels = np.zeros(len(mesh.vertices), dtype=int)

    target_part_id = args.part_id if args.part_id is not None else 0
    drag_point = args.drag_point
    drag_vector = args.drag_vector

    if args.interactive:
        part_id, drag_point, drag_vector = launch_drag_picker(
            mesh=mesh,
            vertex_labels=vertex_labels,
            target_label=target_part_id,
            drag_point_default=drag_point,
            drag_vector_default=drag_vector,
            args=args
        )
        target_part_id = part_id
    else:
        if drag_point is None or drag_vector is None:
            raise ValueError("Provide --drag_point and --drag_vector or use --interactive.")

    part = create_segmented_part(mesh, vertex_labels, target_part_id)
    if part is None:
        raise RuntimeError(f"Failed to extract part {target_part_id}.")
    mask_bool = part.vertex_mask
    part.drag_point = drag_point
    part.drag_vector = drag_vector
    part.drag_point_source = "interactive" if args.interactive else "manual"

    kpp_model = load_kpp_model(args.kpp_checkpoint, device)
    vae_model, num_frames = load_vae_model(args.vae_checkpoint, device, args.num_frames)

    pc, face_indices, center, scale = sample_point_cloud(mesh, args.num_points)
    sampled_mask = np.zeros(len(face_indices), dtype=np.float32)
    for i, fid in enumerate(face_indices):
        face_vertices = mesh.faces[fid]
        sampled_mask[i] = 1.0 if mask_bool[face_vertices].mean() > 0.5 else 0.0

    drag_point_norm = (drag_point - center) / scale
    drag_vector_norm = drag_vector / scale

    pred_type_logits, pred_axis, pred_origin = run_kpp_inference(
        kpp_model,
        pc,
        sampled_mask,
        drag_point_norm,
        drag_vector_norm,
        device
    )
    kpp_joint_type = int(torch.argmax(pred_type_logits, dim=-1).item()) if pred_type_logits is not None else 0
    joint_axis = pred_axis.squeeze(0).cpu().numpy()
    joint_axis = joint_axis / (np.linalg.norm(joint_axis) + 1e-8)
    joint_origin_norm = pred_origin.squeeze(0).cpu().numpy()
    joint_origin = joint_origin_norm * scale + center

    final_joint_type = kpp_joint_type
    llm_response = None

    if args.manual_joint_type is not None:
        final_joint_type = 0 if args.manual_joint_type == 'revolute' else 1
        print(f"Manual joint type override: {args.manual_joint_type}")
    elif args.llm_endpoint and requests is not None:
        classifier = LLMJointClassifier(
            endpoint=args.llm_endpoint,
            api_key=args.llm_api_key,
            model=args.llm_model,
            temperature=args.llm_temperature,
            timeout=args.llm_timeout,
            system_prompt=args.llm_system_prompt,
            max_retries=args.llm_max_retries,
            proxy=args.llm_proxy
        )
        llm_type, llm_response = classifier.classify(
            f"Part {target_part_id} with drag vector {drag_vector_norm.tolist()}."
        )
        if llm_type is not None:
            final_joint_type = llm_type
    elif args.llm_endpoint and requests is None:
        print("requests not installed; skipping LLM.")

    sample_dict = {
        'initial_mesh': mesh,
        'part_mask': mask_bool.astype(np.float32),
        'drag_point': drag_point,
        'drag_vector': drag_vector,
        'joint_type': 'revolute' if final_joint_type == 0 else 'prismatic',
        'joint_axis': joint_axis,
        'joint_origin': joint_origin,
        'rotation_direction': None,
        'trajectory_vectors': None,
        'drag_trajectory': None
    }

    run_animation_from_sample(
        model=vae_model,
        sample=sample_dict,
        sample_name=f"part_{target_part_id}",
        device=device,
        output_root=args.output_dir,
        num_frames=num_frames,
        num_samples_to_gen=args.num_samples,
        force_rotation=False,
        include_groundtruth=False
    )


if __name__ == '__main__':
    main()
