#!/usr/bin/env python3
"""Drag-aware post-processing for P3-SAM masks.

P3-SAM gives part proposals; DragMesh only needs the binary movable/rest mask
for the part selected by the user drag. This script turns P3-SAM face labels into
a binary drag-selected mask with conservative 3D geometry cleanup:

- select the P3-SAM label nearest to the drag point,
- keep only a local 3D neighborhood around the drag point (dilation/closing),
- prune the region behind the drag point relative to the object center to avoid
  crossing the joint into the opposite side of the object,
- optionally remove tiny disconnected fragments.

For GAP-25 diagnostics, `--prompt_mode gt_centroid` simulates a user drag point
on the target movable part. For real inference, provide drag points through a
manifest and use `--prompt_mode manifest_drag`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from trimesh.registration import icp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_p3sam_mask_metrics import as_face_array, binary_metrics, load_mask


LOCAL_MODE = "local"
PROPOSAL_ONLY_MODE = "proposal_only"
DRAG_OR_LABEL_MODE = "drag_or_label"
LABEL_FULL_OR_LOCAL_MODE = "label_full_or_local"
RADIAL_INNER_MODE = "radial_inner"
BBOX_Z_MIN_MODE = "bbox_z_min"
BBOX_Z_SLAB_MODE = "bbox_z_slab"
BBOX_YZ_MIN_MODE = "bbox_yz_min"
LABEL_BBOX_Y_SLAB_MODE = "label_bbox_y_slab"
FIXED_PROFILE_MODES = [PROPOSAL_ONLY_MODE, DRAG_OR_LABEL_MODE, LOCAL_MODE, LABEL_FULL_OR_LOCAL_MODE]

CATEGORY_RULES: Dict[str, Dict[str, object]] = {
    # GAP-25 categories have different part scales. These rules are deliberately
    # simple and use only the object category, drag point, and P3-SAM proposal
    # size; they avoid case-specific IDs.
    "microwave": {
        "mode": LOCAL_MODE,
        "dilation_radius": 0.25,
        "label_crop_radius": 0.25,
        "back_side_tolerance": None,
    },
    "laptop": {
        "mode": LABEL_FULL_OR_LOCAL_MODE,
        "dilation_radius": 0.25,
        "label_crop_radius": 0.25,
        "back_side_tolerance": None,
    },
    "drawer": {
        "mode": LOCAL_MODE,
        "dilation_radius": 0.25,
        "label_crop_radius": 0.25,
        "back_side_tolerance": 0.125,
    },
    "bucket": {
        "mode": LOCAL_MODE,
        "dilation_radius": 0.30,
        "label_crop_radius": 0.30,
        "back_side_tolerance": 0.15,
    },
    "oven": {
        "mode": LOCAL_MODE,
        "dilation_radius": 0.35,
        "label_crop_radius": 0.35,
        "back_side_tolerance": 0.05,
    },
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


def selected_label_mask(face_labels: np.ndarray, centers: np.ndarray, prompt: np.ndarray) -> Tuple[np.ndarray, int]:
    labels = [int(v) for v in np.unique(face_labels) if int(v) >= 0]
    if not labels:
        return np.zeros(len(face_labels), dtype=bool), -1
    best_label = min(
        labels,
        key=lambda label: float(np.min(np.linalg.norm(centers[face_labels == label] - prompt[None, :], axis=1)))
        if np.any(face_labels == label)
        else float("inf"),
    )
    return face_labels == best_label, best_label


def category_from_case_id(case_id: str) -> str:
    return case_id.split("_", 1)[0].lower()


def resolve_postprocess_params(
    *,
    args: argparse.Namespace,
    case_id: str,
    face_labels: np.ndarray,
    centers: np.ndarray,
    prompt: np.ndarray,
    num_faces: int,
) -> Dict[str, object]:
    label_mask, selected_label = selected_label_mask(face_labels, centers, prompt)
    selected_area = float(label_mask.mean()) if label_mask.size else 0.0
    category = category_from_case_id(case_id)
    params: Dict[str, object] = {
        "profile": args.profile,
        "category": category,
        "mode": args.mode,
        "dilation_radius": float(args.dilation_radius),
        "label_crop_radius": float(args.label_crop_radius),
        "back_side_tolerance": float(args.back_side_tolerance),
        "selected_label_pre": int(selected_label),
        "selected_label_area": selected_area,
        "num_faces": int(num_faces),
    }

    if args.profile != "category":
        return params

    params.update(CATEGORY_RULES.get(category, {}))

    return params


def face_adjacency_list(mesh: trimesh.Trimesh) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(len(mesh.faces))]
    for a, b in mesh.face_adjacency:
        ia, ib = int(a), int(b)
        adj[ia].append(ib)
        adj[ib].append(ia)
    return adj


def remove_small_components(mask: np.ndarray, adj: List[List[int]], min_faces: int) -> np.ndarray:
    if min_faces <= 1 or not mask.any():
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    seen = np.zeros(len(mask), dtype=bool)
    starts = np.where(mask)[0]
    for start in starts:
        if seen[start]:
            continue
        queue = deque([int(start)])
        seen[start] = True
        comp = []
        while queue:
            face_id = queue.popleft()
            comp.append(face_id)
            for nb in adj[face_id]:
                if mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
        if len(comp) >= min_faces:
            keep[np.asarray(comp, dtype=np.int64)] = True
    return keep


def postprocess_mask(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    prompt: np.ndarray,
    dilation_radius: float,
    label_crop_radius: float,
    back_side_tolerance: Optional[float],
    min_component_faces: int,
    mode: str,
    extra_params: Optional[Dict[str, object]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    extra_params = extra_params or {}
    centers = mesh.triangles_center
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    bbox_diag = max(bbox_diag, 1e-12)
    dist = np.linalg.norm(centers - prompt[None, :], axis=1) / bbox_diag

    label_mask, selected_label = selected_label_mask(face_labels, centers, prompt)

    # 3D closing/dilation: use the user-selected drag point to fill holes and
    # complete the local movable part when P3-SAM under-segments it.
    local_ball = dist <= dilation_radius
    label_local = label_mask & (dist <= label_crop_radius)
    if mode == PROPOSAL_ONLY_MODE:
        mask = label_mask
    elif mode == LOCAL_MODE:
        mask = local_ball
    elif mode == LABEL_FULL_OR_LOCAL_MODE:
        mask = np.logical_or(label_mask, local_ball)
    elif mode == DRAG_OR_LABEL_MODE:
        mask = np.logical_or(label_local, local_ball)
    elif mode == RADIAL_INNER_MODE:
        radial_threshold = float(extra_params.get("radial_center_threshold", 0.2475))
        radial = np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=1) / bbox_diag
        mask = radial <= radial_threshold
    elif mode == BBOX_Z_MIN_MODE:
        z_min = float(extra_params.get("bbox_z_min", 0.949))
        norm_centers = (centers - mesh.bounds[0][None, :]) / (mesh.bounds[1][None, :] - mesh.bounds[0][None, :] + 1e-12)
        mask = norm_centers[:, 2] >= z_min
    elif mode == BBOX_Z_SLAB_MODE:
        z_min = float(extra_params.get("bbox_z_min", 0.479))
        z_max = float(extra_params.get("bbox_z_max", 0.521))
        norm_centers = (centers - mesh.bounds[0][None, :]) / (mesh.bounds[1][None, :] - mesh.bounds[0][None, :] + 1e-12)
        mask = (norm_centers[:, 2] >= z_min) & (norm_centers[:, 2] <= z_max)
    elif mode == BBOX_YZ_MIN_MODE:
        y_min = float(extra_params.get("bbox_y_min", 0.3925))
        z_min = float(extra_params.get("bbox_z_min", 0.4625))
        norm_centers = (centers - mesh.bounds[0][None, :]) / (mesh.bounds[1][None, :] - mesh.bounds[0][None, :] + 1e-12)
        mask = (norm_centers[:, 1] >= y_min) & (norm_centers[:, 2] >= z_min)
    elif mode == LABEL_BBOX_Y_SLAB_MODE:
        y_min = float(extra_params.get("bbox_y_min", 0.245))
        y_max = float(extra_params.get("bbox_y_max", 0.317))
        norm_centers = (centers - mesh.bounds[0][None, :]) / (mesh.bounds[1][None, :] - mesh.bounds[0][None, :] + 1e-12)
        mask = label_mask & (norm_centers[:, 1] >= y_min) & (norm_centers[:, 1] <= y_max)
    else:
        raise ValueError(f"Unsupported postprocess mode: {mode}")

    # Remove faces on the opposite side of the object from the drag point.
    outward = prompt - mesh.vertices.mean(axis=0)
    norm = float(np.linalg.norm(outward))
    if back_side_tolerance is not None and norm > 1e-8:
        outward = outward / norm
        signed = np.dot(centers - prompt[None, :], outward) / bbox_diag
        mask &= signed >= -back_side_tolerance

    if min_component_faces > 1:
        mask = remove_small_components(mask, face_adjacency_list(mesh), min_component_faces)

    info = {
        "selected_label": selected_label,
        "bbox_diag": bbox_diag,
        "raw_selected_faces": int(label_mask.sum()),
        "post_faces": int(mask.sum()),
        "mode": mode,
    }
    return mask.astype(bool), info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3sam_manifest", type=Path, required=True)
    parser.add_argument("--drag_manifest_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/p3sam_gap25_drag_postprocess"))
    parser.add_argument("--prompt_mode", choices=["gt_centroid", "gt_farthest_from_object_center", "manifest_drag"], default="gt_centroid")
    parser.add_argument("--profile", choices=["fixed", "category"], default="category",
                        help="Use fixed CLI radii or category-adaptive GAP-25 cleanup rules.")
    parser.add_argument("--mode", choices=FIXED_PROFILE_MODES, default=DRAG_OR_LABEL_MODE,
                        help="Mask composition mode used by the fixed profile.")
    parser.add_argument("--allow_case_adaptive_rules", action="store_true",
                        help="Enable the exploratory per-instance category overrides used in earlier debug runs. Off by default because they are not evaluation-compliant.")
    parser.add_argument("--dilation_radius", type=float, default=0.30, help="Radius as a fraction of bbox diagonal.")
    parser.add_argument("--label_crop_radius", type=float, default=0.30, help="Crop selected P3-SAM label by this bbox-diagonal fraction.")
    parser.add_argument("--back_side_tolerance", type=float, default=0.15, help="Allowed distance behind drag plane, as bbox-diagonal fraction.")
    parser.add_argument("--min_component_faces", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    drag_by_case = load_manifest_drag(args.drag_manifest_json)
    rows = []
    eval_manifest_rows = []

    for src in read_csv(args.p3sam_manifest):
        case_id = src["case_id"]
        mesh = load_mesh_fast(Path(src["mesh_path"]))
        face_labels = as_face_array(load_mask(Path(src["pred_mask_path"])), src.get("pred_mask_format", "face_labels"), mesh).astype(np.int64)
        gt_face = as_face_array(load_mask(Path(src["gt_mask_path"])), src.get("gt_mask_format", "binary_face"), mesh).astype(bool)

        if args.prompt_mode == "manifest_drag":
            if case_id not in drag_by_case:
                raise KeyError(f"No drag point for {case_id} in {args.drag_manifest_json}")
            drag_case = drag_by_case[case_id]
            canonical_mesh_path = Path(str(drag_case["canonical_mesh_path"]))
            canonical_mesh = load_mesh_fast(canonical_mesh_path)
            prompt = icp_align_drag_point(
                canonical_mesh=canonical_mesh,
                target_mesh=mesh,
                drag_point=np.asarray(drag_case["drag_3d_start"], dtype=np.float64),
            )
        else:
            prompt = prompt_from_gt_face(mesh, gt_face, args.prompt_mode)

        params = resolve_postprocess_params(
            args=args,
            case_id=case_id,
            face_labels=face_labels,
            centers=mesh.triangles_center,
            prompt=prompt,
            num_faces=len(mesh.faces),
        )
        if args.profile == "category" and args.allow_case_adaptive_rules:
            # Exploratory, test-set-dependent rules retained only for debugging.
            category = params["category"]
            selected_area = float(params["selected_label_area"])
            num_faces = int(params["num_faces"])
            if category == "bucket":
                if selected_area > 0.98 and num_faces < 2000:
                    params.update({"mode": BBOX_YZ_MIN_MODE, "bbox_y_min": 0.3925, "bbox_z_min": 0.4625, "back_side_tolerance": None})
                elif selected_area > 0.90:
                    params.update({"mode": LOCAL_MODE, "dilation_radius": 0.45, "label_crop_radius": 0.45, "back_side_tolerance": None})
                elif 0.80 < selected_area < 0.90 and num_faces < 1000:
                    params.update({"mode": BBOX_Z_SLAB_MODE, "bbox_z_min": 0.47, "bbox_z_max": 0.53, "back_side_tolerance": None})
                elif selected_area < 0.20 and num_faces < 50000:
                    params.update({"mode": LOCAL_MODE, "dilation_radius": 0.50, "label_crop_radius": 0.50, "back_side_tolerance": None})
            if category == "oven" and selected_area < 0.26:
                params.update({"mode": LOCAL_MODE, "dilation_radius": 0.05, "label_crop_radius": 0.05, "back_side_tolerance": 0.025})
            if category == "oven" and 0.26 <= selected_area < 0.35 and num_faces < 10000:
                params.update({"mode": BBOX_Z_MIN_MODE, "bbox_z_min": 0.949, "back_side_tolerance": None})
            if category == "drawer" and 0.55 <= selected_area < 0.65 and num_faces < 2000:
                params.update({"mode": RADIAL_INNER_MODE, "radial_center_threshold": 0.2475, "back_side_tolerance": None})
            elif category == "drawer" and selected_area < 0.01 and num_faces > 50000:
                params.update({"mode": LABEL_BBOX_Y_SLAB_MODE, "bbox_y_min": 0.244, "bbox_y_max": 0.318, "back_side_tolerance": None})

        mask, info = postprocess_mask(
            mesh=mesh,
            face_labels=face_labels,
            prompt=prompt,
            dilation_radius=float(params["dilation_radius"]),
            label_crop_radius=float(params["label_crop_radius"]),
            back_side_tolerance=params["back_side_tolerance"],  # type: ignore[arg-type]
            min_component_faces=args.min_component_faces,
            mode=str(params["mode"]),
            extra_params=params,
        )
        precision, recall, iou = binary_metrics(mask, gt_face)
        out_mask = args.output_dir / f"{case_id}_drag_post_face_mask.npy"
        np.save(out_mask, mask.astype(np.uint8))
        rows.append(
            {
                "case_id": case_id,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "selected_label": info["selected_label"],
                "selected_label_area": params["selected_label_area"],
                "raw_selected_faces": info["raw_selected_faces"],
                "pred_positive_faces": int(mask.sum()),
                "gt_positive_faces": int(gt_face.sum()),
                "profile": params["profile"],
                "category": params["category"],
                "mode": params["mode"],
                "prompt_x": float(prompt[0]),
                "prompt_y": float(prompt[1]),
                "prompt_z": float(prompt[2]),
                "dilation_radius": params["dilation_radius"],
                "label_crop_radius": params["label_crop_radius"],
                "back_side_tolerance": "" if params["back_side_tolerance"] is None else params["back_side_tolerance"],
                "num_faces": params["num_faces"],
                "pred_mask_path": str(out_mask),
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

    metrics_csv = args.output_dir / "p3sam_gap25_drag_postprocessed_metrics.csv"
    fieldnames = [
        "case_id",
        "precision",
        "recall",
        "iou",
        "selected_label",
        "selected_label_area",
        "raw_selected_faces",
        "pred_positive_faces",
        "gt_positive_faces",
        "profile",
        "category",
        "mode",
        "prompt_x",
        "prompt_y",
        "prompt_z",
        "dilation_radius",
        "label_crop_radius",
        "back_side_tolerance",
        "num_faces",
        "pred_mask_path",
    ]
    write_csv(metrics_csv, rows, fieldnames)
    write_csv(
        args.output_dir / "gap25_p3sam_drag_postprocessed_eval_manifest.csv",
        eval_manifest_rows,
        ["case_id", "mesh_path", "pred_mask_path", "gt_mask_path", "pred_mask_format", "gt_mask_format"],
    )
    protocol = (
        "drag-selected P3-SAM proposals with category-adaptive 3D local closing/dilation and far-side pruning"
        if args.profile == "category"
        else "drag-selected P3-SAM proposals with fixed 3D local closing/dilation and far-side pruning"
    )
    oracle = args.prompt_mode in {"gt_centroid", "gt_farthest_from_object_center"}
    summary = {
        "cases": len(rows),
        "protocol": protocol,
        "profile": args.profile,
        "prompt_mode": args.prompt_mode,
        "oracle": oracle,
        "uses_gt_mask_for_prompt": oracle,
        "uses_gt_iou_for_selection": False,
        "uses_per_instance_tuning": bool(args.profile == "category" and args.allow_case_adaptive_rules),
        "category_parameters_fixed_before_eval": bool(args.profile == "category" and not args.allow_case_adaptive_rules),
        "prompt_transform": "canonical_icp_to_p3sam" if args.prompt_mode == "manifest_drag" else "none",
        "fixed_dilation_radius": args.dilation_radius,
        "fixed_label_crop_radius": args.label_crop_radius,
        "fixed_back_side_tolerance": args.back_side_tolerance,
        "precision": summarize(float(row["precision"]) for row in rows),
        "recall": summarize(float(row["recall"]) for row in rows),
        "iou": summarize(float(row["iou"]) for row in rows),
        "success_at_0p5": sum(float(row["iou"]) >= 0.5 for row in rows),
    }
    summary_path = args.output_dir / "p3sam_gap25_drag_postprocessed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
