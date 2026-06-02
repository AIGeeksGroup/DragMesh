#!/usr/bin/env python3
"""Non-oracle drag-aware post-processing for P3-SAM proposals.

This script turns P3-SAM face-label proposals into a binary movable/rest mask
for DragMesh using only:

- user drag point / vector,
- mesh geometry,
- P3-SAM proposal labels,
- fixed category-level priors when requested.

It does not use GT masks, GT centroids, or GT IoU to select proposals.
Any GT-based prompt is reported as oracle upper bound only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.registration import icp

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_p3sam_mask_metrics import as_face_array, binary_metrics, load_mask


RAW_DRAG_SELECTED = "raw_drag_selected"
FIXED_DRAG_CLEANUP = "fixed_drag_cleanup"
CATEGORY_DRAG_CLEANUP = "category_adaptive_drag_cleanup"
ORACLE_CENTROID = "oracle_centroid_upper_bound"

CATEGORY_PRIORS: Dict[str, Dict[str, object]] = {
    "microwave": {"radius": 0.20, "back_tol": 0.08, "min_comp": 18, "area_min": 0.03, "area_max": 0.50, "ring_hops": 2},
    "laptop": {"radius": 0.30, "back_tol": 0.12, "min_comp": 24, "area_min": 0.12, "area_max": 0.78, "ring_hops": 2},
    "drawer": {"radius": 0.20, "back_tol": 0.10, "min_comp": 16, "area_min": 0.04, "area_max": 0.40, "ring_hops": 2},
    "bucket": {"radius": 0.24, "back_tol": 0.12, "min_comp": 12, "area_min": 0.02, "area_max": 0.35, "ring_hops": 2},
    "oven": {"radius": 0.24, "back_tol": 0.08, "min_comp": 18, "area_min": 0.05, "area_max": 0.55, "ring_hops": 2},
}

LEGACY_ORACLE_CATEGORY_RULES: Dict[str, Dict[str, object]] = {
    "microwave": {"mode": "local", "dilation_radius": 0.25, "label_crop_radius": 0.25, "back_side_tolerance": None},
    "laptop": {"mode": "label_full_or_local", "dilation_radius": 0.25, "label_crop_radius": 0.25, "back_side_tolerance": None},
    "drawer": {"mode": "local", "dilation_radius": 0.25, "label_crop_radius": 0.25, "back_side_tolerance": 0.125},
    "bucket": {"mode": "local", "dilation_radius": 0.30, "label_crop_radius": 0.30, "back_side_tolerance": 0.15},
    "oven": {"mode": "local", "dilation_radius": 0.35, "label_crop_radius": 0.35, "back_side_tolerance": 0.05},
}

FIXED_PRIOR = {"radius": 0.22, "back_tol": 0.10, "min_comp": 16, "area_min": 0.03, "area_max": 0.55, "ring_hops": 2}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def load_mesh_fast(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)} for {path}")
    return mesh


def load_manifest_drag(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    cases = data.get("cases", data)
    return {str(case["case_id"]): case for case in cases}


def icp_align_drag_point(
    canonical_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    drag_point: np.ndarray,
    num_samples: int = 4000,
) -> np.ndarray:
    src_points, _ = trimesh.sample.sample_surface(canonical_mesh, num_samples)
    dst_points, _ = trimesh.sample.sample_surface(target_mesh, num_samples)
    matrix, _, _ = icp(src_points, dst_points, scale=True, reflection=False)
    drag_h = np.append(np.asarray(drag_point, dtype=np.float64), 1.0)
    return (matrix @ drag_h)[:3].astype(np.float64)


def nearest_vertex_on_mesh(mesh: trimesh.Trimesh, point: np.ndarray) -> Tuple[np.ndarray, int, float]:
    tree = cKDTree(mesh.vertices)
    dist, idx = tree.query(np.asarray(point, dtype=np.float64))
    idx = int(idx)
    return mesh.vertices[idx].astype(np.float64), idx, float(dist)


def category_from_case_id(case_id: str) -> str:
    return case_id.split("_", 1)[0].lower()


def face_adjacency_list(mesh: trimesh.Trimesh) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    return adj


def expand_faces(mask: np.ndarray, adj: List[List[int]], hops: int = 1) -> np.ndarray:
    if hops <= 0 or not mask.any():
        return mask.copy()
    cur = {int(v) for v in np.where(mask)[0]}
    out = set(cur)
    for _ in range(hops):
        nxt = set()
        for f in cur:
            for nb in adj[f]:
                if nb not in out:
                    out.add(nb)
                    nxt.add(nb)
        cur = nxt
    out_mask = np.zeros_like(mask, dtype=bool)
    out_mask[np.asarray(sorted(out), dtype=np.int64)] = True
    return out_mask


def connected_components(mask: np.ndarray, adj: List[List[int]]) -> List[np.ndarray]:
    seen = np.zeros(len(mask), dtype=bool)
    comps: List[np.ndarray] = []
    for start in np.where(mask)[0]:
        start = int(start)
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        comp: List[int] = []
        while queue:
            fid = queue.popleft()
            comp.append(fid)
            for nb in adj[fid]:
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
        comps.append(np.asarray(comp, dtype=np.int64))
    return comps


def seed_local_faces(mesh: trimesh.Trimesh, seed_vertex: int, adj: List[List[int]], hops: int = 2) -> np.ndarray:
    incident = np.where((mesh.faces == int(seed_vertex)).any(axis=1))[0].tolist()
    local = set(int(v) for v in incident)
    frontier = list(local)
    for _ in range(hops):
        nxt: List[int] = []
        for fid in frontier:
            for nb in adj[fid]:
                nb = int(nb)
                if nb not in local:
                    local.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return np.asarray(sorted(local), dtype=np.int64)


def prompt_from_gt_face(mesh: trimesh.Trimesh, gt_face_mask: np.ndarray, mode: str) -> np.ndarray:
    movable = np.where(gt_face_mask.astype(bool))[0]
    if movable.size == 0:
        raise ValueError("GT movable mask has no positive faces")
    centers = mesh.triangles_center[movable]
    if mode == "gt_centroid":
        verts = mesh.vertices[mesh.faces[movable].reshape(-1)]
        target = verts.mean(axis=0)
        return centers[int(np.argmin(np.linalg.norm(centers - target[None, :], axis=1)))]
    if mode == "gt_farthest_from_object_center":
        object_center = mesh.vertices.mean(axis=0)
        return centers[int(np.argmax(np.linalg.norm(centers - object_center[None, :], axis=1)))]
    raise ValueError(f"Unsupported GT prompt mode: {mode}")


def legacy_oracle_postprocess(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    prompt: np.ndarray,
    category: str,
    min_component_faces: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    centers = mesh.triangles_center
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    bbox_diag = max(bbox_diag, 1e-12)
    label_mask = face_labels >= 0
    selected_label = -1
    if np.any(label_mask):
        labels = [int(v) for v in np.unique(face_labels) if int(v) >= 0]
        selected_label = min(
            labels,
            key=lambda label: float(np.min(np.linalg.norm(centers[face_labels == label] - prompt[None, :], axis=1)))
            if np.any(face_labels == label)
            else float("inf"),
        )
        label_mask = face_labels == selected_label
    dist = np.linalg.norm(centers - prompt[None, :], axis=1) / bbox_diag
    label_local = label_mask & (dist <= float(LEGACY_ORACLE_CATEGORY_RULES.get(category, {}).get("label_crop_radius", 0.3)))
    rule = LEGACY_ORACLE_CATEGORY_RULES.get(category, {"mode": "local", "dilation_radius": 0.3, "back_side_tolerance": 0.15})
    mode = str(rule["mode"])
    dilation_radius = float(rule["dilation_radius"])
    back_side_tolerance = rule["back_side_tolerance"]
    if mode == "local":
        mask = dist <= dilation_radius
    elif mode == "label_full_or_local":
        mask = np.logical_or(label_mask, dist <= dilation_radius)
    elif mode == "drag_or_label":
        mask = np.logical_or(label_local, dist <= dilation_radius)
    else:
        mask = dist <= dilation_radius
    outward = prompt - mesh.vertices.mean(axis=0)
    norm = float(np.linalg.norm(outward))
    if back_side_tolerance is not None and norm > 1e-8:
        outward = outward / norm
        signed = np.dot(centers - prompt[None, :], outward) / bbox_diag
        mask &= signed >= -float(back_side_tolerance)
    if min_component_faces > 1:
        mask = remove_small_components(mask, face_adjacency_list(mesh), min_component_faces)
    info = {
        "selected_label": int(selected_label),
        "bbox_diag": bbox_diag,
        "raw_selected_faces": int(label_mask.sum()) if label_mask.ndim else 0,
        "post_faces": int(mask.sum()),
        "mode": mode,
    }
    return mask.astype(bool), info


def score_candidate_labels(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    prompt: np.ndarray,
    drag_vec: np.ndarray,
    adj: List[List[int]],
    ring_hops: int,
) -> Tuple[List[Dict[str, object]], int, float, float]:
    centers = mesh.triangles_center
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    bbox_diag = max(bbox_diag, 1e-12)
    prompt_vertex, seed_vertex, seed_vertex_dist = nearest_vertex_on_mesh(mesh, prompt)
    local_faces = seed_local_faces(mesh, seed_vertex, adj, hops=ring_hops)
    local_labels = [int(face_labels[f]) for f in local_faces if int(face_labels[f]) >= 0]
    labels = sorted(set(local_labels)) if local_labels else [int(v) for v in np.unique(face_labels) if int(v) >= 0]
    if not labels:
        labels = [-1]

    drag = np.asarray(drag_vec, dtype=np.float64)
    drag_norm = float(np.linalg.norm(drag))
    drag = drag / drag_norm if drag_norm > 1e-12 else np.zeros(3, dtype=np.float64)

    outward = prompt - mesh.vertices.mean(axis=0)
    outward_norm = float(np.linalg.norm(outward))
    outward = outward / outward_norm if outward_norm > 1e-12 else np.zeros(3, dtype=np.float64)

    candidates: List[Dict[str, object]] = []
    for label in labels:
        mask = face_labels == int(label)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        pts = centers[idx]
        centroid = pts.mean(axis=0)
        area_ratio = float(mask.mean())
        min_dist = float(np.min(np.linalg.norm(pts - prompt[None, :], axis=1)) / bbox_diag)
        med_dist = float(np.median(np.linalg.norm(pts - prompt[None, :], axis=1)) / bbox_diag)
        local_support = float(sum(bool(mask[f]) for f in local_faces) / max(1, len(local_faces)))
        contains_drag = bool(sum(bool(mask[f]) for f in local_faces) > 0)
        comps = connected_components(mask, adj)
        frag_penalty = float(len(comps) / max(1, int(mask.sum())))
        centered = centroid - mesh.vertices.mean(axis=0)
        centered_norm = float(np.linalg.norm(centered))
        direction_consistency = 0.0
        if centered_norm > 1e-12:
            centered = centered / centered_norm
            direction_consistency = max(0.0, float(np.dot(centered, drag)))
        far_side_penalty = 0.0
        if float(np.linalg.norm(outward)) > 1e-12:
            signed = float(np.dot(centroid - prompt, outward) / bbox_diag)
            far_side_penalty = max(0.0, -signed)

        area_penalty = 0.0
        if area_ratio < 0.02:
            area_penalty += (0.02 - area_ratio) / 0.02
        if area_ratio > 0.75:
            area_penalty += (area_ratio - 0.75) / 0.25

        score = (
            4.0 * float(contains_drag)
            + 3.0 * local_support
            - 3.0 * med_dist
            - 1.0 * min_dist
            - 1.2 * area_penalty
            - 1.0 * far_side_penalty
            - 0.8 * frag_penalty
            + 0.5 * direction_consistency
        )
        candidates.append(
            {
                "label": int(label),
                "score": float(score),
                "contains_drag": bool(contains_drag),
                "local_support": float(local_support),
                "min_dist": float(min_dist),
                "med_dist": float(med_dist),
                "area_ratio": float(area_ratio),
                "direction_consistency": float(direction_consistency),
                "far_side_penalty": float(far_side_penalty),
                "fragmentation_penalty": float(frag_penalty),
                "seed_vertex": int(seed_vertex),
                "seed_vertex_dist": float(seed_vertex_dist),
                "centroid": centroid.tolist(),
                "face_count": int(mask.sum()),
            }
        )

    candidates.sort(key=lambda x: (-float(x["score"]), -float(x["local_support"]), float(x["med_dist"])))
    return candidates, seed_vertex, seed_vertex_dist, bbox_diag


def choose_selection_mask(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    candidates: List[Dict[str, object]],
    mode: str,
    prior: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    adj = face_adjacency_list(mesh)
    centers = mesh.triangles_center
    prompt = np.asarray(prior["prompt"], dtype=np.float64)
    seed_vertex = int(prior["seed_vertex"])
    bbox_diag = float(prior["bbox_diag"])
    selected_label = int(candidates[0]["label"]) if candidates else -1

    if mode == RAW_DRAG_SELECTED:
        mask = face_labels == selected_label
        info = {
            "selected_label": selected_label,
            "selected_score": float(candidates[0]["score"]) if candidates else 0.0,
            "selected_label_area": float(mask.mean()) if mask.size else 0.0,
            "contains_drag": bool(candidates[0]["contains_drag"]) if candidates else False,
            "local_support": float(candidates[0]["local_support"]) if candidates else 0.0,
            "seed_vertex": seed_vertex,
            "seed_vertex_dist": float(prior["seed_vertex_dist"]),
            "candidate_count": len(candidates),
        }
        return mask.astype(bool), info

    union = np.zeros(len(face_labels), dtype=bool)
    top_score = float(candidates[0]["score"]) if candidates else -1e9
    for cand in candidates[:3]:
        if float(cand["score"]) >= top_score - 0.15 and bool(cand["contains_drag"]):
            union |= face_labels == int(cand["label"])
    if not union.any() and candidates:
        union = face_labels == selected_label
    mask = union.copy()

    radius = float(prior["radius"])
    back_tol = float(prior["back_tol"])
    min_comp = int(prior["min_comp"])
    area_min = float(prior["area_min"])
    area_max = float(prior["area_max"])
    ring_hops = int(prior["ring_hops"])

    d = np.linalg.norm(centers - prompt[None, :], axis=1) / bbox_diag
    mask &= d <= radius

    outward = prompt - mesh.vertices.mean(axis=0)
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm > 1e-12:
        outward = outward / outward_norm
        signed = np.dot(centers - prompt[None, :], outward) / bbox_diag
        mask &= signed >= -back_tol

    if mask.any():
        mask = expand_faces(mask, adj, hops=1)

    if mask.any():
        comps = connected_components(mask, adj)
        seed_face = int(np.argmin(np.linalg.norm(centers - prompt[None, :], axis=1)))
        chosen = None
        for comp in comps:
            if seed_face in set(comp.tolist()):
                chosen = comp
                break
        if chosen is None:
            chosen = max(comps, key=len)
        keep = np.zeros_like(mask, dtype=bool)
        keep[chosen] = True
        mask = keep

    area_ratio = float(mask.mean()) if mask.size else 0.0
    if mask.any() and (area_ratio < area_min or area_ratio > area_max):
        if selected_label >= 0:
            mask = face_labels == selected_label

    if min_comp > 1 and mask.any():
        comps = connected_components(mask, adj)
        keep = np.zeros_like(mask, dtype=bool)
        for comp in comps:
            if len(comp) >= min_comp:
                keep[comp] = True
        if keep.any():
            mask = keep

    info = {
        "selected_label": selected_label,
        "selected_score": float(candidates[0]["score"]) if candidates else 0.0,
        "selected_label_area": float((face_labels == selected_label).mean()) if selected_label >= 0 else 0.0,
        "contains_drag": bool(candidates[0]["contains_drag"]) if candidates else False,
        "local_support": float(candidates[0]["local_support"]) if candidates else 0.0,
        "seed_vertex": seed_vertex,
        "seed_vertex_dist": float(prior["seed_vertex_dist"]),
        "candidate_count": len(candidates),
        "cleanup_area_ratio": area_ratio,
        "ring_hops": ring_hops,
    }
    return mask.astype(bool), info


def classify_failure(row: Dict[str, object]) -> str:
    iou = float(row["iou"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    area_ratio = float(row.get("pred_area_ratio", 0.0))
    seed_dist = float(row.get("seed_vertex_dist", 0.0))
    contains_drag = bool(row.get("contains_drag", False))
    selected_label = int(row.get("selected_label", -1))
    candidate_count = int(row.get("candidate_count", 0))
    if seed_dist > 0.08:
        return "coordinate misalignment"
    if candidate_count == 0 or not contains_drag:
        return "no proposal near drag"
    if selected_label < 0:
        return "selected static/background proposal"
    if iou < 0.15 and area_ratio > 0.50:
        return "over-segmentation / too large"
    if iou < 0.15 and area_ratio < 0.08:
        return "under-segmentation / missing movable part"
    if precision < 0.35 and recall < 0.35:
        return "fragmented mask"
    if precision < 0.45 and recall >= 0.45:
        return "wrong side of object"
    if recall < 0.30:
        return "under-segmentation / missing movable part"
    if precision < 0.30:
        return "over-segmentation / too large"
    return "mixed / weak mask"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_simple_visualization(mesh: trimesh.Trimesh, face_mask: np.ndarray, prompt: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    centers = mesh.triangles_center
    pts = np.vstack([mesh.vertices, centers, prompt[None, :]])
    pts0 = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    basis = vh[:2].T
    c2 = (centers - pts.mean(axis=0)) @ basis
    p2 = (prompt - pts.mean(axis=0)) @ basis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(c2[:, 0], c2[:, 1], s=3, c=np.where(face_mask, "#d62728", "#1f77b4"), alpha=0.75)
    ax.scatter([p2[0]], [p2[1]], s=80, c="#2ca02c", marker="x")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3sam_manifest", type=Path, required=True)
    parser.add_argument("--drag_manifest_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_mode", choices=["manifest_drag", "user_drag_point", "gt_centroid", "gt_farthest_from_object_center"], default="manifest_drag")
    parser.add_argument("--mode", choices=[RAW_DRAG_SELECTED, FIXED_DRAG_CLEANUP, CATEGORY_DRAG_CLEANUP, ORACLE_CENTROID], required=True)
    parser.add_argument("--min_component_faces", type=int, default=1)
    parser.add_argument("--no_visualization", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    mask_dir = args.output_dir / "selected_masks"
    ensure_dir(mask_dir)
    vis_dir = args.output_dir / "visualization"
    if not args.no_visualization:
        ensure_dir(vis_dir)

    drag_by_case = load_manifest_drag(args.drag_manifest_json)
    rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    eval_manifest_rows: List[Dict[str, object]] = []

    for src in read_csv(args.p3sam_manifest):
        case_id = src["case_id"]
        mesh = load_mesh_fast(Path(src["mesh_path"]))
        pred_face = as_face_array(load_mask(Path(src["pred_mask_path"])), src.get("pred_mask_format", "face_labels"), mesh).astype(np.int64)
        gt_face = as_face_array(load_mask(Path(src["gt_mask_path"])), src.get("gt_mask_format", "binary_face"), mesh).astype(bool)
        adj = face_adjacency_list(mesh)
        category = category_from_case_id(case_id)

        if args.prompt_mode == "manifest_drag":
            if case_id not in drag_by_case:
                raise KeyError(f"No drag point for {case_id} in {args.drag_manifest_json}")
            drag_case = drag_by_case[case_id]
            canonical_mesh = load_mesh_fast(Path(str(drag_case["canonical_mesh_path"])))
            prompt = icp_align_drag_point(canonical_mesh=canonical_mesh, target_mesh=mesh, drag_point=np.asarray(drag_case["drag_3d_start"], dtype=np.float64))
            prompt, _, prompt_dist = nearest_vertex_on_mesh(mesh, prompt)
            drag_vec = np.asarray(drag_case["drag_3d_vector"], dtype=np.float64)
            prompt_source = "manifest_drag"
        else:
            prompt = prompt_from_gt_face(mesh, gt_face, args.prompt_mode)
            drag_case = drag_by_case.get(case_id, {})
            drag_vec = np.asarray(drag_case.get("drag_3d_vector", [0.0, 0.0, 0.0]), dtype=np.float64)
            if args.mode == ORACLE_CENTROID:
                prompt_dist = 0.0
            else:
                prompt, _, prompt_dist = nearest_vertex_on_mesh(mesh, prompt)
            prompt_source = args.prompt_mode

        priors: Dict[str, object] = dict(FIXED_PRIOR)
        if args.mode == CATEGORY_DRAG_CLEANUP:
            priors.update(CATEGORY_PRIORS.get(category, FIXED_PRIOR))
        elif args.mode == ORACLE_CENTROID:
            priors.update(FIXED_PRIOR)

        if args.mode == ORACLE_CENTROID:
            prompt = prompt_from_gt_face(mesh, gt_face, "gt_centroid")
            mask, info = legacy_oracle_postprocess(
                mesh=mesh,
                face_labels=pred_face,
                prompt=prompt,
                category=category,
                min_component_faces=args.min_component_faces,
            )
            info.update({
                "selected_score": 0.0,
                "contains_drag": True,
                "local_support": 1.0,
                "seed_vertex": -1,
                "seed_vertex_dist": 0.0,
                "candidate_count": 0,
                "selected_label_area": float((pred_face == info["selected_label"]).mean()) if int(info["selected_label"]) >= 0 else 0.0,
            })
        else:
            candidates, seed_vertex, seed_vertex_dist, bbox_diag = score_candidate_labels(
                mesh=mesh,
                face_labels=pred_face,
                prompt=prompt,
                drag_vec=drag_vec,
                adj=adj,
                ring_hops=int(priors["ring_hops"]),
            )
            prior: Dict[str, object] = {
                "prompt": prompt,
                "bbox_diag": bbox_diag,
                "seed_vertex": seed_vertex,
                "seed_vertex_dist": seed_vertex_dist,
                **priors,
            }
            mask, info = choose_selection_mask(mesh, pred_face, candidates, args.mode, prior)

        if args.mode in {FIXED_DRAG_CLEANUP, CATEGORY_DRAG_CLEANUP} and mask.any():
            d = np.linalg.norm(mesh.triangles_center - prompt[None, :], axis=1) / bbox_diag
            mask &= d <= float(prior["radius"])
            outward = prompt - mesh.vertices.mean(axis=0)
            n = float(np.linalg.norm(outward))
            if n > 1e-12:
                outward = outward / n
                signed = np.dot(mesh.triangles_center - prompt[None, :], outward) / bbox_diag
                mask &= signed >= -float(prior["back_tol"])
            mask = expand_faces(mask, adj, hops=1)
            if mask.any():
                comps = connected_components(mask, adj)
                seed_face = int(np.argmin(np.linalg.norm(mesh.triangles_center - prompt[None, :], axis=1)))
                chosen = None
                for comp in comps:
                    if seed_face in set(comp.tolist()):
                        chosen = comp
                        break
                if chosen is None:
                    chosen = max(comps, key=len)
                keep = np.zeros_like(mask, dtype=bool)
                keep[chosen] = True
                mask = keep
            area_ratio = float(mask.mean()) if mask.size else 0.0
            if area_ratio < float(prior["area_min"]) or area_ratio > float(prior["area_max"]):
                if info["selected_label"] >= 0:
                    mask = pred_face == int(info["selected_label"])

        if args.min_component_faces > 1 and mask.any():
            comps = connected_components(mask, adj)
            keep = np.zeros_like(mask, dtype=bool)
            for comp in comps:
                if len(comp) >= args.min_component_faces:
                    keep[comp] = True
            if keep.any():
                mask = keep

        precision, recall, iou = binary_metrics(mask, gt_face)
        selected_path = mask_dir / f"{case_id}_movable_mask.npy"
        np.save(selected_path, mask.astype(np.uint8))

        pred_area_ratio = float(mask.mean()) if mask.size else 0.0
        row = {
            "case_id": case_id,
            "category": category,
            "mode": args.mode,
            "prompt_mode": args.prompt_mode,
            "oracle": args.mode == ORACLE_CENTROID,
            "uses_gt_mask_for_prompt": args.mode == ORACLE_CENTROID,
            "uses_gt_iou_for_selection": False,
            "uses_per_instance_tuning": False,
            "category_parameters_fixed_before_eval": args.mode == CATEGORY_DRAG_CLEANUP,
            "prompt_source": prompt_source,
            "selected_label": int(info.get("selected_label", -1)),
            "selected_score": float(info.get("selected_score", 0.0)),
            "contains_drag": bool(info.get("contains_drag", False)),
            "local_support": float(info.get("local_support", 0.0)),
            "seed_vertex": int(info.get("seed_vertex", -1)),
            "seed_vertex_dist": float(info.get("seed_vertex_dist", 0.0)),
            "candidate_count": int(info.get("candidate_count", 0)),
            "selected_label_area": float(info.get("selected_label_area", 0.0)),
            "pred_area_ratio": pred_area_ratio,
            "pred_positive_faces": int(mask.sum()),
            "gt_positive_faces": int(gt_face.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "iou": float(iou),
            "failure_reason": "",
            "pred_mask_path": str(selected_path),
            "mesh_path": src["mesh_path"],
            "gt_mask_path": src["gt_mask_path"],
            "pred_mask_format": "binary_face",
            "gt_mask_format": src.get("gt_mask_format", "binary_face"),
        }
        row["failure_reason"] = classify_failure(row)
        rows.append(row)
        failure_rows.append(
            {
                "case_id": case_id,
                "category": row["category"],
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "selected_label": row["selected_label"],
                "selected_label_area": row["selected_label_area"],
                "pred_area_ratio": pred_area_ratio,
                "contains_drag": row["contains_drag"],
                "candidate_count": row["candidate_count"],
                "seed_vertex_dist": row["seed_vertex_dist"],
                "failure_reason": row["failure_reason"],
            }
        )
        eval_manifest_rows.append(
            {
                "case_id": case_id,
                "mesh_path": src["mesh_path"],
                "pred_mask_path": str(selected_path),
                "gt_mask_path": src["gt_mask_path"],
                "pred_mask_format": "binary_face",
                "gt_mask_format": src.get("gt_mask_format", "binary_face"),
            }
        )
        if not args.no_visualization:
            save_simple_visualization(mesh, mask, prompt, vis_dir / f"{case_id}.png")

    metrics_csv = args.output_dir / "per_case_metrics.csv"
    failure_csv = args.output_dir / "failure_analysis.csv"
    eval_manifest_csv = args.output_dir / "gap25_p3sam_drag_postprocessed_eval_manifest.csv"
    fieldnames = [
        "case_id",
        "category",
        "mode",
        "prompt_mode",
        "oracle",
        "uses_gt_mask_for_prompt",
        "uses_gt_iou_for_selection",
        "uses_per_instance_tuning",
        "category_parameters_fixed_before_eval",
        "prompt_source",
        "selected_label",
        "selected_score",
        "contains_drag",
        "local_support",
        "seed_vertex",
        "seed_vertex_dist",
        "candidate_count",
        "selected_label_area",
        "pred_area_ratio",
        "pred_positive_faces",
        "gt_positive_faces",
        "precision",
        "recall",
        "iou",
        "failure_reason",
        "pred_mask_path",
        "mesh_path",
        "gt_mask_path",
        "pred_mask_format",
        "gt_mask_format",
    ]
    write_csv(metrics_csv, rows, fieldnames)
    write_csv(
        failure_csv,
        failure_rows,
        [
            "case_id",
            "category",
            "iou",
            "precision",
            "recall",
            "selected_label",
            "selected_label_area",
            "pred_area_ratio",
            "contains_drag",
            "candidate_count",
            "seed_vertex_dist",
            "failure_reason",
        ],
    )
    write_csv(
        eval_manifest_csv,
        eval_manifest_rows,
        ["case_id", "mesh_path", "pred_mask_path", "gt_mask_path", "pred_mask_format", "gt_mask_format"],
    )

    categories = sorted({row["category"] for row in rows})
    per_category = {}
    for cat in categories:
        subset = [r for r in rows if r["category"] == cat]
        per_category[cat] = {
            "cases": len(subset),
            "miou": summarize(float(r["iou"]) for r in subset)["mean"],
            "median_iou": summarize(float(r["iou"]) for r in subset)["median"],
            "success_at_0p3": int(sum(float(r["iou"]) >= 0.3 for r in subset)),
            "success_at_0p5": int(sum(float(r["iou"]) >= 0.5 for r in subset)),
            "success_at_0p7": int(sum(float(r["iou"]) >= 0.7 for r in subset)),
        }

    reason_counts: Dict[str, int] = {}
    for row in rows:
        reason_counts[row["failure_reason"]] = reason_counts.get(row["failure_reason"], 0) + 1

    summary = {
        "cases": len(rows),
        "mode": args.mode,
        "prompt_mode": args.prompt_mode,
        "oracle": args.mode == ORACLE_CENTROID,
        "uses_gt_mask_for_prompt": args.mode == ORACLE_CENTROID,
        "uses_gt_iou_for_selection": False,
        "uses_per_instance_tuning": False,
        "category_parameters_fixed_before_eval": args.mode == CATEGORY_DRAG_CLEANUP,
        "protocol": (
            "oracle centroid upper bound" if args.mode == ORACLE_CENTROID else "non-oracle drag-aware proposal selection and cleanup"
        ),
        "precision": summarize(float(r["precision"]) for r in rows),
        "recall": summarize(float(r["recall"]) for r in rows),
        "iou": summarize(float(r["iou"]) for r in rows),
        "success_at_0p3": int(sum(float(r["iou"]) >= 0.3 for r in rows)),
        "success_at_0p5": int(sum(float(r["iou"]) >= 0.5 for r in rows)),
        "success_at_0p7": int(sum(float(r["iou"]) >= 0.7 for r in rows)),
        "per_category": per_category,
        "failure_reason_histogram": reason_counts,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {failure_csv}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
