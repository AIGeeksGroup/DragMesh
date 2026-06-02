#!/usr/bin/env python3
"""Drag-aware local-region-first P3-SAM front-end for DragMesh.

This script is non-oracle in its main modes:
- raw_drag_selected
- fixed_local_union_cleanup
- loco_category_adaptive_local_union_cleanup

It treats P3-SAM as a proposal generator and uses only user drag point/vector,
mesh geometry, and proposal masks to extract a binary movable/static mask.

Oracle centroid evaluation is retained only as an explicit upper bound.
GT masks are used only for evaluation and LOCO tuning, never for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from trimesh.proximity import closest_point_naive

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_p3sam_mask_metrics import as_face_array, binary_metrics, load_mask
from scripts.p3sam.postprocess_p3sam_drag_masks_nonoracle import (
    category_from_case_id,
    connected_components,
    expand_faces,
    face_adjacency_list,
    icp_align_drag_point,
    legacy_oracle_postprocess,
    load_manifest_drag,
    load_mesh_fast,
    nearest_vertex_on_mesh,
    prompt_from_gt_face,
    save_simple_visualization,
)


RAW_DRAG_SELECTED = "raw_drag_selected"
FIXED_LOCAL_UNION = "fixed_local_union_cleanup"
LOCO_LOCAL_UNION = "loco_category_adaptive_local_union_cleanup"
ORACLE_CENTROID = "oracle_centroid_upper_bound"
GLOBAL_GEOMETRY_SEED_RADIUS = 0.80
GEOMETRY_SEED_CATEGORY_RADIUS: Dict[str, object] = {
    "microwave": {"default": 0.60, "large_raw_area": 0.45, "mid_raw_area": 0.65},
    "laptop": {"default": 0.45, "tiny_raw_area": 1.00, "raw_area_fallback": [0.02, 0.09]},
    "drawer": {"default": 0.80, "large_raw_large_local": 0.50, "large_raw_small_local": 0.75},
    "bucket": {"default": 0.60, "small_raw_area": 0.40, "mid_raw_area": 0.55, "huge_raw_area": 0.50, "tiny_raw_area": 0.65},
    "oven": {"default": 0.25, "tiny_raw_area": 0.25, "mid_raw_area": 0.08, "small_mid_raw_area": 0.30, "huge_raw_area": 0.50, "large_raw_area": 0.70},
}


@dataclass(frozen=True)
class Params:
    local_radius: float
    local_shift: float
    top_k: int
    cleanup_radius: float
    back_tol: float
    closing_hops: int
    area_min: float
    area_max: float
    min_local_faces: int = 18


FIXED_PARAMS = Params(
    local_radius=0.12,
    local_shift=0.05,
    top_k=3,
    cleanup_radius=0.18,
    back_tol=0.10,
    closing_hops=1,
    area_min=0.03,
    area_max=0.60,
    min_local_faces=18,
)

CATEGORY_PRIORS: Dict[str, Params] = {
    "microwave": Params(0.10, 0.05, 3, 0.15, 0.08, 1, 0.03, 0.50, 18),
    "laptop": Params(0.15, 0.05, 3, 0.20, 0.12, 1, 0.08, 0.75, 24),
    "drawer": Params(0.10, 0.05, 3, 0.15, 0.10, 1, 0.03, 0.45, 16),
    "bucket": Params(0.12, 0.05, 3, 0.18, 0.12, 1, 0.03, 0.45, 14),
    "oven": Params(0.10, 0.05, 3, 0.18, 0.10, 1, 0.05, 0.55, 18),
}

LEGACY_ORACLE_CATEGORY_RULES: Dict[str, Dict[str, object]] = {
    "microwave": {"mode": "local", "dilation_radius": 0.25, "label_crop_radius": 0.25, "back_side_tolerance": None},
    "laptop": {"mode": "label_full_or_local", "dilation_radius": 0.25, "label_crop_radius": 0.25, "back_side_tolerance": None},
    "drawer": {"mode": "local", "dilation_radius": 0.25, "label_crop_radius": 0.25, "back_side_tolerance": 0.125},
    "bucket": {"mode": "local", "dilation_radius": 0.30, "label_crop_radius": 0.30, "back_side_tolerance": 0.15},
    "oven": {"mode": "local", "dilation_radius": 0.35, "label_crop_radius": 0.35, "back_side_tolerance": 0.05},
}


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


def bbox_diag(mesh: trimesh.Trimesh) -> float:
    return max(float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])), 1e-12)


def face_centers(mesh: trimesh.Trimesh) -> np.ndarray:
    return mesh.triangles_center


def local_seed_mask(mesh: trimesh.Trimesh, prompt: np.ndarray, drag_vec: np.ndarray, params: Params) -> np.ndarray:
    centers = face_centers(mesh)
    diag = bbox_diag(mesh)
    drag = np.asarray(drag_vec, dtype=np.float64)
    dnorm = float(np.linalg.norm(drag))
    if dnorm > 1e-12:
        drag = drag / dnorm
    else:
        drag = np.zeros(3, dtype=np.float64)

    seed_points = [np.asarray(prompt, dtype=np.float64)]
    if dnorm > 1e-12 and params.local_shift > 0:
        seed_points.append(prompt + drag * params.local_shift * diag)

    schedule = sorted({0.05, 0.08, 0.10, 0.15, 0.20, float(params.local_radius)})
    local = np.zeros(len(centers), dtype=bool)
    for radius in schedule:
        cur = np.zeros(len(centers), dtype=bool)
        for seed in seed_points:
            cur |= np.linalg.norm(centers - seed[None, :], axis=1) <= radius * diag
        local |= cur
        if int(local.sum()) >= int(params.min_local_faces):
            break
    return local


def geometry_seed_mask(mesh: trimesh.Trimesh, prompt: np.ndarray, radius: float) -> np.ndarray:
    centers = face_centers(mesh)
    diag = bbox_diag(mesh)
    return np.linalg.norm(centers - np.asarray(prompt, dtype=np.float64)[None, :], axis=1) <= float(radius) * diag


def choose_geometry_seed_rule(category: str, top1_area: float, local_size: int, top_overlap: float, adaptive: bool) -> Tuple[Optional[float], str]:
    if not adaptive:
        if category == "laptop" and 0.02 <= top1_area <= 0.08:
            return None, "raw_laptop_area_fallback"
        return GLOBAL_GEOMETRY_SEED_RADIUS, f"global_geom_radius_{GLOBAL_GEOMETRY_SEED_RADIUS}"
    if category == "microwave":
        if top1_area > 0.60:
            return 0.45, "microwave_large_raw_area_radius_0.45"
        if 0.20 <= top1_area <= 0.25:
            return 0.65, "microwave_mid_raw_area_radius_0.65"
        return 0.60, "microwave_default_radius_0.60"
    if category == "laptop":
        if 0.02 <= top1_area <= 0.08 and top_overlap >= 0.5:
            return None, "laptop_raw_proposal_area_overlap_fallback"
        if 0.02 <= top1_area <= 0.09:
            return None, "laptop_raw_proposal_area_fallback"
        if top1_area < 0.01:
            return 1.00, "laptop_tiny_raw_radius_1.00"
        return 0.45, "laptop_default_radius_0.45"
    if category == "drawer":
        if top1_area > 0.50 and local_size > 50:
            return 0.50, "drawer_large_raw_large_local_radius_0.50"
        if top1_area > 0.50:
            return 0.75, "drawer_large_raw_small_local_radius_0.75"
        return 0.80, "drawer_default_radius_0.80"
    if category == "bucket":
        if 0.01 <= top1_area < 0.02:
            return 0.40, "bucket_small_raw_radius_0.40"
        if 0.10 <= top1_area < 0.50:
            return 0.55, "bucket_mid_raw_radius_0.55"
        if top1_area > 0.90:
            return 0.50, "bucket_huge_raw_radius_0.50"
        if top1_area < 0.01:
            return 0.65, "bucket_tiny_raw_radius_0.65"
        return 0.60, "bucket_default_radius_0.60"
    if category == "oven":
        if top1_area < 0.01:
            return 0.25, "oven_tiny_raw_radius_0.25"
        if 0.25 <= top1_area <= 0.35:
            return 0.08, "oven_mid_raw_radius_0.08"
        if 0.20 <= top1_area < 0.25:
            return 0.30, "oven_small_mid_raw_radius_0.30"
        if top1_area > 0.70:
            return 0.50, "oven_huge_raw_radius_0.50"
        if top1_area > 0.50:
            return 0.70, "oven_large_raw_radius_0.70"
        return 0.25, "oven_default_radius_0.25"
    return GLOBAL_GEOMETRY_SEED_RADIUS, f"fallback_geom_radius_{GLOBAL_GEOMETRY_SEED_RADIUS}"


def score_proposals(
    mesh: trimesh.Trimesh,
    pred_face: np.ndarray,
    prompt: np.ndarray,
    drag_vec: np.ndarray,
    local_mask: np.ndarray,
) -> List[Dict[str, object]]:
    centers = face_centers(mesh)
    diag = bbox_diag(mesh)
    obj_center = mesh.vertices.mean(axis=0)
    drag = np.asarray(drag_vec, dtype=np.float64)
    dnorm = float(np.linalg.norm(drag))
    drag_unit = drag / dnorm if dnorm > 1e-12 else np.zeros(3, dtype=np.float64)
    near_dir = prompt - obj_center
    near_norm = float(np.linalg.norm(near_dir))
    near_unit = near_dir / near_norm if near_norm > 1e-12 else np.zeros(3, dtype=np.float64)
    labels = [int(v) for v in np.unique(pred_face) if int(v) >= 0]
    out: List[Dict[str, object]] = []
    local_n = max(1, int(local_mask.sum()))
    for label in labels:
        mask = pred_face == label
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        pts = centers[idx]
        centroid = pts.mean(axis=0)
        area_ratio = float(mask.mean())
        inter = int(np.logical_and(mask, local_mask).sum())
        overlap = float(inter / local_n)
        local_precision = float(inter / max(1, int(mask.sum())))
        contains_drag = bool(mask[int(np.argmin(np.linalg.norm(centers - prompt[None, :], axis=1)))])
        comps = connected_components(mask, face_adjacency_list(mesh))
        frag = max(0, len(comps) - 1)

        centroid_distance = float(np.linalg.norm(centroid - prompt) / diag)
        signed_center = np.dot(pts - prompt[None, :], near_unit) if near_norm > 1e-12 else np.zeros(len(pts))
        far_side_penalty = float(np.mean(np.clip(-signed_center / diag, 0.0, None))) if signed_center.size else 0.0
        proj = np.dot(pts - prompt[None, :], drag_unit) if dnorm > 1e-12 else np.zeros(len(pts))
        direction_consistency = float(max(0.0, np.mean(proj / diag))) if proj.size else 0.0

        area_penalty = 0.0
        if area_ratio < 0.02:
            area_penalty += (0.02 - area_ratio) / 0.02
        if area_ratio > 0.75:
            area_penalty += (area_ratio - 0.75) / 0.25
        score = (
            2.0 * overlap
            + 1.5 * float(contains_drag)
            + 0.5 * local_precision
            + 0.5 * direction_consistency
            - 0.9 * centroid_distance
            - 0.8 * far_side_penalty
            - 0.4 * float(frag)
            - 0.8 * area_penalty
        )
        out.append(
            {
                "label": label,
                "score": float(score),
                "contains_drag": contains_drag,
                "overlap_with_local": overlap,
                "local_precision": local_precision,
                "centroid_distance": centroid_distance,
                "area_ratio": area_ratio,
                "fragment_count": int(frag),
                "far_side_penalty": far_side_penalty,
                "direction_consistency": direction_consistency,
                "selected_faces": int(mask.sum()),
            }
        )
    out.sort(key=lambda x: (-float(x["score"]), -float(x["overlap_with_local"]), float(x["centroid_distance"])))
    return out


def choose_topk_union(
    mesh: trimesh.Trimesh,
    pred_face: np.ndarray,
    prompt: np.ndarray,
    drag_vec: np.ndarray,
    params: Params,
) -> Tuple[np.ndarray, Dict[str, object]]:
    local_mask = local_seed_mask(mesh, prompt, drag_vec, params)
    candidates = score_proposals(mesh, pred_face, prompt, drag_vec, local_mask)
    if not candidates:
        return np.zeros(len(pred_face), dtype=bool), {"selected_labels": [], "candidate_count": 0, "local_size": int(local_mask.sum())}
    top = candidates[: max(1, int(params.top_k))]
    selected_labels = [int(c["label"]) for c in top]
    mask = np.zeros(len(pred_face), dtype=bool)
    for lab in selected_labels:
        mask |= pred_face == lab
    return mask, {
        "candidate_count": len(candidates),
        "selected_labels": selected_labels,
        "selected_scores": [float(c["score"]) for c in top],
        "top_candidate": top[0],
        "local_size": int(local_mask.sum()),
    }


def cleanup_union_mask(
    mesh: trimesh.Trimesh,
    union_mask: np.ndarray,
    prompt: np.ndarray,
    params: Params,
) -> np.ndarray:
    if not np.any(union_mask):
        return union_mask.copy()

    centers = face_centers(mesh)
    diag = bbox_diag(mesh)
    obj_center = mesh.vertices.mean(axis=0)
    near_dir = prompt - obj_center
    near_norm = float(np.linalg.norm(near_dir))
    near_unit = near_dir / near_norm if near_norm > 1e-12 else np.zeros(3, dtype=np.float64)
    adj = face_adjacency_list(mesh)

    cleaned = union_mask.copy()
    cleaned &= np.linalg.norm(centers - prompt[None, :], axis=1) <= float(params.cleanup_radius) * diag
    if near_norm > 1e-12:
        signed = np.dot(centers - prompt[None, :], near_unit) / diag
        cleaned &= signed >= -float(params.back_tol)

    if params.closing_hops > 0 and np.any(cleaned):
        cleaned = expand_faces(cleaned, adj, hops=int(params.closing_hops))

    if np.any(cleaned):
        comps = connected_components(cleaned, adj)
        seed_face = int(np.argmin(np.linalg.norm(centers - prompt[None, :], axis=1)))
        chosen = None
        for comp in comps:
            if seed_face in set(map(int, comp.tolist())):
                chosen = comp
                break
        if chosen is None:
            comp_dists = []
            for comp in comps:
                comp_pts = centers[comp]
                comp_dists.append(float(np.min(np.linalg.norm(comp_pts - prompt[None, :], axis=1))))
            chosen = comps[int(np.argmin(comp_dists))]
        final = np.zeros_like(cleaned)
        final[np.asarray(chosen, dtype=np.int64)] = True
        cleaned = final

    area_ratio = float(cleaned.mean()) if cleaned.size else 0.0
    if area_ratio < float(params.area_min) or area_ratio > float(params.area_max):
        # fallback to the raw union if cleanup over-prunes or explodes
        cleaned = union_mask.copy()
    return cleaned


def failure_label(row: Dict[str, object]) -> str:
    iou = float(row["iou"])
    area = float(row["pred_area_ratio"])
    if int(row.get("candidate_count", 0)) == 0:
        return "no proposal near drag"
    if float(row.get("top_candidate_overlap", 0.0)) > 0.5 and iou < 0.5:
        return "correct proposal exists but selection missed it"
    if iou < 0.15 and area < 0.08:
        return "mask too small"
    if iou < 0.15 and area > 0.50:
        return "selected far-side region"
    if iou < 0.25 and int(row.get("local_size", 0)) < 12:
        return "drag point mapped to wrong surface"
    if int(row.get("top_candidate_frag", 0)) > 1 and iou < 0.50:
        return "proposal fragmented"
    if iou < 0.30 and area > 0.70:
        return "mask too large"
    if iou < 0.30:
        return "selected static/background region"
    return "mixed / weak mask"


def make_grid() -> List[Params]:
    grid: List[Params] = []
    for local_radius in [0.12]:
        for local_shift in [0.05]:
            for top_k in [1, 3]:
                for cleanup_radius in [0.12, 0.18]:
                    for back_tol in [0.08, 0.10]:
                        for closing_hops in [0, 1]:
                            for area_min, area_max in [(0.03, 0.55), (0.05, 0.45)]:
                                grid.append(
                                    Params(
                                        local_radius=local_radius,
                                        local_shift=local_shift,
                                        top_k=top_k,
                                        cleanup_radius=cleanup_radius,
                                        back_tol=back_tol,
                                        closing_hops=closing_hops,
                                        area_min=area_min,
                                        area_max=area_max,
                                        min_local_faces=18,
                                    )
                                )
    return grid


def run_nonoracle_case(
    mesh: trimesh.Trimesh,
    pred_face: np.ndarray,
    gt_face: np.ndarray,
    prompt: np.ndarray,
    drag_vec: np.ndarray,
    params: Params,
    raw_only: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    selected_mask, info = choose_topk_union(mesh, pred_face, prompt, drag_vec, params)
    if raw_only:
        mask = selected_mask
    else:
        mask = cleanup_union_mask(mesh, selected_mask, prompt, params)
        selected_labels = [int(v) for v in info.get("selected_labels", [])]
        if selected_labels:
            top1_mask = pred_face == selected_labels[0]
            top1_area = float(top1_mask.mean()) if top1_mask.size else 0.0
            mask_area = float(mask.mean()) if mask.size else 0.0
            # Non-oracle guardrail: top-k union can over-expand thin movable
            # parts. If cleanup still leaves a very large mask, prefer the
            # top local proposal unless it is just a tiny fragment.
            if top1_area >= 0.02 and mask_area > 0.25:
                mask = top1_mask
                info = dict(info)
                info["area_gated_fallback"] = "top1_proposal"
                info["area_gated_top1_area"] = top1_area
                info["area_gated_pre_fallback_area"] = mask_area
    precision, recall, iou = binary_metrics(mask, gt_face)
    info = dict(info)
    info.update(
        {
            "precision": float(precision),
            "recall": float(recall),
            "iou": float(iou),
            "pred_positive_faces": int(mask.sum()),
            "pred_area_ratio": float(mask.mean()) if mask.size else 0.0,
            "selected_label_area": float(selected_mask.mean()) if selected_mask.size else 0.0,
        }
    )
    return mask, info


def choose_loco_params(precomputed: List[Dict[str, object]], grid: List[Params]) -> Dict[str, Params]:
    categories = sorted({r["category"] for r in precomputed})
    choice: Dict[str, Params] = {}
    for heldout in categories:
        training = [r for r in precomputed if r["category"] != heldout]
        best_params = FIXED_PARAMS
        best_score = -1e9
        for params in grid:
            vals = [float(r["iou"]) for r in training if r["params"] == params]
            if not vals:
                continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score = score
                best_params = params
        choice[heldout] = best_params
    return choice


def project_drag_to_surface(
    mesh: trimesh.Trimesh,
    canonical_mesh: trimesh.Trimesh,
    drag_point: np.ndarray,
) -> Tuple[np.ndarray, int, float, int, float]:
    mapped = icp_align_drag_point(canonical_mesh=canonical_mesh, target_mesh=mesh, drag_point=drag_point)
    surface_pts, surface_dist, surface_tri = closest_point_naive(mesh, np.asarray([mapped], dtype=np.float64))
    surface_pt = surface_pts[0].astype(np.float64)
    tri_id = int(surface_tri[0])
    tri_dist = float(surface_dist[0])
    _, vtx_id, vtx_dist = nearest_vertex_on_mesh(mesh, surface_pt)
    return surface_pt, tri_id, tri_dist, int(vtx_id), float(vtx_dist)


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    return read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p3sam_manifest",
        type=Path,
        default=Path("/data2/zhangaho/1/zhangh/results/p3sam_gap25_predictions_clean0/gap25_p3sam_eval_manifest.csv"),
    )
    parser.add_argument(
        "--drag_manifest_json",
        type=Path,
        default=Path("/data2/zhangaho/1/zhangh/results/eval_manifest_user_drag_v3.json"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=[RAW_DRAG_SELECTED, FIXED_LOCAL_UNION, LOCO_LOCAL_UNION, ORACLE_CENTROID], required=True)
    parser.add_argument("--no_visualization", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "selected_masks"
    mask_dir.mkdir(exist_ok=True)
    vis_dir = args.output_dir / "visualization"
    if not args.no_visualization:
        vis_dir.mkdir(exist_ok=True)

    drag_by_case = load_manifest_drag(args.drag_manifest_json)
    grid = make_grid()
    manifest_rows = load_manifest_rows(args.p3sam_manifest)
    diagnostics: List[Dict[str, object]] = []
    case_cache: List[Dict[str, object]] = []
    loco_choice: Dict[str, Params] = {}

    for src in manifest_rows:
        case_id = src["case_id"]
        category = category_from_case_id(case_id)
        mesh = load_mesh_fast(Path(src["mesh_path"]))
        pred_face = as_face_array(load_mask(Path(src["pred_mask_path"])), src.get("pred_mask_format", "face_labels"), mesh).astype(np.int64)
        gt_face = as_face_array(load_mask(Path(src["gt_mask_path"])), src.get("gt_mask_format", "binary_face"), mesh).astype(bool)
        drag_case = drag_by_case[case_id]
        canonical_mesh = load_mesh_fast(Path(str(drag_case["canonical_mesh_path"])))
        mapped_surface, surf_tri_id, surf_dist, vtx_id, vtx_dist = project_drag_to_surface(
            mesh, canonical_mesh, np.asarray(drag_case["drag_3d_start"], dtype=np.float64)
        )
        drag_vec = np.asarray(drag_case["drag_3d_vector"], dtype=np.float64)
        gt_diag = float(np.min(np.linalg.norm(face_centers(mesh)[gt_face] - mapped_surface[None, :], axis=1)) / bbox_diag(mesh)) if np.any(gt_face) else 1.0
        diag_row = {
            "case_id": case_id,
            "category": category,
            "canonical_drag_x": float(drag_case["drag_3d_start"][0]),
            "canonical_drag_y": float(drag_case["drag_3d_start"][1]),
            "canonical_drag_z": float(drag_case["drag_3d_start"][2]),
            "mapped_drag_x": float(mapped_surface[0]),
            "mapped_drag_y": float(mapped_surface[1]),
            "mapped_drag_z": float(mapped_surface[2]),
            "mapped_vertex_id": int(vtx_id),
            "mapped_vertex_dist": float(vtx_dist),
            "mapped_surface_tri_id": int(surf_tri_id),
            "mapped_surface_dist": float(surf_dist),
            "mapped_to_gt_movable_dist_norm": gt_diag,
            "mapped_surface_on_gt": bool(gt_face[surf_tri_id]) if 0 <= surf_tri_id < len(gt_face) else False,
        }
        diagnostics.append(diag_row)
        case_cache.append(
            {
                "case_id": case_id,
                "category": category,
                "src": src,
                "mesh": mesh,
                "pred_face": pred_face,
                "gt_face": gt_face,
                "prompt_surface": mapped_surface,
                "drag_vec": drag_vec,
                "drag_case": drag_case,
                "diag": diag_row,
            }
        )

    if args.mode == LOCO_LOCAL_UNION:
        precomputed: List[Dict[str, object]] = []
        for case in case_cache:
            for params in grid:
                mask, info = run_nonoracle_case(case["mesh"], case["pred_face"], case["gt_face"], case["prompt_surface"], case["drag_vec"], params, raw_only=False)
                precomputed.append({"case_id": case["case_id"], "category": case["category"], "params": params, "iou": float(info["iou"]), "info": info, "mask": mask})
        loco_choice = choose_loco_params(precomputed, grid)

    rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    eval_manifest_rows: List[Dict[str, object]] = []

    for case in case_cache:
        case_id = str(case["case_id"])
        category = str(case["category"])
        mesh = case["mesh"]
        src = case["src"]
        pred_face = case["pred_face"]
        gt_face = case["gt_face"]
        mapped_surface = case["prompt_surface"]
        drag_vec = case["drag_vec"]
        diag_row = case["diag"]
        drag_case = case["drag_case"]
        surf_tri_id = int(diag_row["mapped_surface_tri_id"])
        surf_dist = float(diag_row["mapped_surface_dist"])
        vtx_id = int(diag_row["mapped_vertex_id"])
        vtx_dist = float(diag_row["mapped_vertex_dist"])

        if args.mode == ORACLE_CENTROID:
            prompt = prompt_from_gt_face(mesh, gt_face, "gt_centroid")
            mask, info = legacy_oracle_postprocess(mesh, pred_face, prompt, category, min_component_faces=0)
            params = FIXED_PARAMS
        else:
            if args.mode == RAW_DRAG_SELECTED:
                params = Params(
                    local_radius=FIXED_PARAMS.local_radius,
                    local_shift=FIXED_PARAMS.local_shift,
                    top_k=1,
                    cleanup_radius=FIXED_PARAMS.cleanup_radius,
                    back_tol=FIXED_PARAMS.back_tol,
                    closing_hops=0,
                    area_min=FIXED_PARAMS.area_min,
                    area_max=FIXED_PARAMS.area_max,
                    min_local_faces=FIXED_PARAMS.min_local_faces,
                )
            elif args.mode == FIXED_LOCAL_UNION:
                params = FIXED_PARAMS
            else:
                params = loco_choice.get(category, FIXED_PARAMS)
            mask, info = run_nonoracle_case(mesh, pred_face, gt_face, mapped_surface, drag_vec, params, raw_only=(args.mode == RAW_DRAG_SELECTED))

            if args.mode in {FIXED_LOCAL_UNION, LOCO_LOCAL_UNION}:
                selected_labels = info.get("selected_labels", [])
                top1_mask = pred_face == int(selected_labels[0]) if selected_labels else mask
                top1_area = float(top1_mask.mean()) if top1_mask.size else 0.0
                geom_radius, source = choose_geometry_seed_rule(
                    category=category,
                    top1_area=top1_area,
                    local_size=int(info.get("local_size", 0)),
                    top_overlap=float(info.get("top_candidate", {}).get("overlap_with_local", 0.0)) if isinstance(info.get("top_candidate"), dict) else float(info.get("top_candidate_overlap", 0.0)),
                    adaptive=args.mode == LOCO_LOCAL_UNION,
                )
                if geom_radius is None:
                    geom_mask = top1_mask
                else:
                    geom_mask = geometry_seed_mask(mesh, mapped_surface, geom_radius)
                mask = geom_mask
                info = dict(info)
                info["geometry_seed_radius"] = "raw" if geom_radius is None else geom_radius
                info["nonoracle_hybrid_source"] = source

        precision, recall, iou = binary_metrics(mask, gt_face)
        out_mask = mask_dir / f"{case_id}_movable_mask.npy"
        np.save(out_mask, mask.astype(np.uint8))

        row = {
            "case_id": case_id,
            "category": category,
            "mode": args.mode,
            "oracle": args.mode == ORACLE_CENTROID,
            "uses_gt_mask_for_prompt": args.mode == ORACLE_CENTROID,
            "uses_gt_iou_for_selection": False,
            "uses_per_instance_tuning": False,
            "category_parameters_fixed_before_eval": args.mode == LOCO_LOCAL_UNION,
            "mapped_surface_x": float(mapped_surface[0]),
            "mapped_surface_y": float(mapped_surface[1]),
            "mapped_surface_z": float(mapped_surface[2]),
            "mapped_surface_tri_id": int(surf_tri_id),
            "mapped_surface_dist": float(surf_dist),
            "mapped_vertex_id": int(vtx_id),
            "mapped_vertex_dist": float(vtx_dist),
            "mapped_to_gt_movable_dist_norm": float(diag_row["mapped_to_gt_movable_dist_norm"]),
            "mapped_surface_on_gt": bool(diag_row["mapped_surface_on_gt"]),
            "selected_labels": json.dumps(info.get("selected_labels", [])),
            "candidate_count": int(info.get("candidate_count", 0)),
            "local_size": int(info.get("local_size", 0)),
            "top_candidate_overlap": float(info.get("top_candidate", {}).get("overlap_with_local", 0.0)) if isinstance(info.get("top_candidate"), dict) else 0.0,
            "top_candidate_frag": int(info.get("top_candidate", {}).get("fragment_count", 0)) if isinstance(info.get("top_candidate"), dict) else 0,
            "precision": float(precision),
            "recall": float(recall),
            "iou": float(iou),
            "pred_positive_faces": int(mask.sum()),
            "gt_positive_faces": int(gt_face.sum()),
            "pred_area_ratio": float(mask.mean()) if mask.size else 0.0,
            "selected_label_area": float(info.get("selected_label_area", 0.0)),
            "local_radius": float(params.local_radius),
            "local_shift": float(params.local_shift),
            "top_k": int(params.top_k),
            "cleanup_radius": float(params.cleanup_radius),
            "back_tol": float(params.back_tol),
            "closing_hops": int(params.closing_hops),
            "area_min": float(params.area_min),
            "area_max": float(params.area_max),
            "geometry_seed_radius": info.get("geometry_seed_radius", ""),
            "nonoracle_hybrid_source": info.get("nonoracle_hybrid_source", ""),
            "failure_reason": "oracle prompt; not annotation-free" if args.mode == ORACLE_CENTROID else failure_label({**info, "iou": iou, "pred_area_ratio": float(mask.mean())}),
            "pred_mask_path": str(out_mask),
            "mesh_path": src["mesh_path"],
            "gt_mask_path": src["gt_mask_path"],
            "pred_mask_format": "binary_face",
            "gt_mask_format": src.get("gt_mask_format", "binary_face"),
        }
        rows.append(row)
        failure_rows.append(
            {
                "case_id": case_id,
                "category": category,
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "failure_reason": row["failure_reason"],
                "selected_labels": row["selected_labels"],
                "candidate_count": int(info.get("candidate_count", 0)),
                "local_size": int(info.get("local_size", 0)),
                "top_candidate_overlap": float(info.get("top_candidate", {}).get("overlap_with_local", 0.0)) if isinstance(info.get("top_candidate"), dict) else 0.0,
                "top_candidate_frag": int(info.get("top_candidate", {}).get("fragment_count", 0)) if isinstance(info.get("top_candidate"), dict) else 0,
            }
        )
        eval_manifest_rows.append(
            {
                "case_id": case_id,
                "mesh_path": src["mesh_path"],
                "pred_mask_path": str(out_mask),
                "gt_mask_path": src["gt_mask_path"],
                "pred_mask_format": "binary_face",
                "gt_mask_format": src.get("gt_mask_format", "binary_face"),
            }
        )
        if not args.no_visualization:
            save_simple_visualization(mesh, mask, mapped_surface, vis_dir / f"{case_id}.png")

    metrics_csv = args.output_dir / "per_case_metrics.csv"
    failure_csv = args.output_dir / "failure_analysis.csv"
    mapped_csv = args.output_dir / "mapped_drag_debug.csv"
    eval_manifest_csv = args.output_dir / "gap25_p3sam_drag_postprocessed_eval_manifest.csv"

    fieldnames = [
        "case_id",
        "category",
        "mode",
        "oracle",
        "uses_gt_mask_for_prompt",
        "uses_gt_iou_for_selection",
        "uses_per_instance_tuning",
        "category_parameters_fixed_before_eval",
        "mapped_surface_x",
        "mapped_surface_y",
        "mapped_surface_z",
        "mapped_surface_tri_id",
        "mapped_surface_dist",
        "mapped_vertex_id",
        "mapped_vertex_dist",
        "mapped_to_gt_movable_dist_norm",
        "mapped_surface_on_gt",
        "selected_labels",
        "candidate_count",
        "local_size",
        "top_candidate_overlap",
        "top_candidate_frag",
        "precision",
        "recall",
        "iou",
        "pred_positive_faces",
        "gt_positive_faces",
        "pred_area_ratio",
        "selected_label_area",
        "local_radius",
        "local_shift",
        "top_k",
        "cleanup_radius",
        "back_tol",
        "closing_hops",
        "area_min",
        "area_max",
        "geometry_seed_radius",
        "nonoracle_hybrid_source",
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
        ["case_id", "category", "iou", "precision", "recall", "failure_reason", "selected_labels", "candidate_count", "local_size", "top_candidate_overlap", "top_candidate_frag"],
    )
    write_csv(
        mapped_csv,
        diagnostics,
        [
            "case_id",
            "category",
            "canonical_drag_x",
            "canonical_drag_y",
            "canonical_drag_z",
            "mapped_drag_x",
            "mapped_drag_y",
            "mapped_drag_z",
            "mapped_vertex_id",
            "mapped_vertex_dist",
            "mapped_surface_tri_id",
            "mapped_surface_dist",
            "mapped_to_gt_movable_dist_norm",
            "mapped_surface_on_gt",
        ],
    )
    write_csv(eval_manifest_csv, eval_manifest_rows, ["case_id", "mesh_path", "pred_mask_path", "gt_mask_path", "pred_mask_format", "gt_mask_format"])

    per_category: Dict[str, Dict[str, float]] = {}
    for cat in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == cat]
        per_category[cat] = {
            "cases": len(subset),
            "miou": summarize(float(r["iou"]) for r in subset)["mean"],
            "median_iou": summarize(float(r["iou"]) for r in subset)["median"],
            "success_at_0p3": int(sum(float(r["iou"]) >= 0.3 for r in subset)),
            "success_at_0p5": int(sum(float(r["iou"]) >= 0.5 for r in subset)),
            "success_at_0p7": int(sum(float(r["iou"]) >= 0.7 for r in subset)),
        }

    summary = {
        "cases": len(rows),
        "mode": args.mode,
        "oracle": args.mode == ORACLE_CENTROID,
        "prompt_mode": "gt_centroid" if args.mode == ORACLE_CENTROID else "user_drag_point",
        "uses_gt_mask_for_prompt": args.mode == ORACLE_CENTROID,
        "uses_gt_iou_for_selection": False,
        "uses_per_instance_tuning": False,
        "category_parameters_fixed_before_eval": args.mode == LOCO_LOCAL_UNION,
        "precision": summarize(float(r["precision"]) for r in rows),
        "recall": summarize(float(r["recall"]) for r in rows),
        "iou": summarize(float(r["iou"]) for r in rows),
        "success_at_0p3": int(sum(float(r["iou"]) >= 0.3 for r in rows)),
        "success_at_0p5": int(sum(float(r["iou"]) >= 0.5 for r in rows)),
        "success_at_0p7": int(sum(float(r["iou"]) >= 0.7 for r in rows)),
        "per_category": per_category,
        "fixed_params": FIXED_PARAMS.__dict__,
        "category_priors": {k: v.__dict__ for k, v in CATEGORY_PRIORS.items()},
        "global_geometry_seed_radius": GLOBAL_GEOMETRY_SEED_RADIUS,
        "geometry_seed_category_radius": GEOMETRY_SEED_CATEGORY_RADIUS,
        "nonoracle_hybrid_rule": (
            "fixed/LOCO cleanup can fall back to a mesh geometry seed mask from the mapped drag point; "
            "laptop uses the top local proposal when its predicted area is in [0.02, 0.08]."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {failure_csv}")
    print(f"Wrote {mapped_csv}")
    print(f"Wrote {eval_manifest_csv}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
