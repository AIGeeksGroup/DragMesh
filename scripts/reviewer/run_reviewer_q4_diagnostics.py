#!/usr/bin/env python3
"""Reviewer-Q4 diagnostics for optional foundation-model dependencies.

This script builds two small tables:

1. Optional upstream diagnostics:
   KPP joint-type accuracy, optional GPT/VLM joint-type accuracy from a CSV log,
   and P3-SAM mask mIoU / Success@0.5.
2. Downstream propagation:
   Provided-mask KPP metrics from the locked GAP-25 CSV, optional P3-SAM-mask
   KPP metrics by rerunning KPP with P3-SAM masks, and optional GPT-type metrics
   by replacing the KPP type with GPT/VLM type while keeping KPP axis/origin.

The P3-SAM rerun is intentionally lightweight: it does not call P3-SAM; it
reuses saved P3-SAM face masks and only reruns the local KPP joint predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from inference_pipeline import face_binary_to_vertex_labels, face_to_vertex_labels, load_mask_array


TRUE_VALUES = {"1", "true", "t", "yes", "y"}
JOINT_ALIASES = {
    "revolute": "revolute",
    "continuous": "revolute",
    "rotate": "revolute",
    "rotation": "revolute",
    "hinge": "revolute",
    "prismatic": "prismatic",
    "slider": "prismatic",
    "slide": "prismatic",
    "linear": "prismatic",
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


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    return float(text)


def canonical_joint_type(value: object) -> Optional[str]:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in JOINT_ALIASES:
        return JOINT_ALIASES[text]
    for key, val in JOINT_ALIASES.items():
        if key in text:
            return val
    return None


def fmt_ratio(numer: int, denom: int) -> str:
    return f"{numer}/{denom}" if denom else "N/A"


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    return "N/A"


def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"mean": None, "median": None, "max": None, "n": 0}
    return {"mean": mean(vals), "median": median(vals), "max": max(vals), "n": len(vals)}


def load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(mesh)}")
    return mesh


def load_manifest_cases(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text())
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    return {str(case["case_id"]): case for case in cases}


def axis_error_deg(pred_axis: np.ndarray, gt_axis: np.ndarray) -> float:
    pred = pred_axis.astype(np.float64)
    gt = gt_axis.astype(np.float64)
    pred = pred / (np.linalg.norm(pred) + 1e-12)
    gt = gt / (np.linalg.norm(gt) + 1e-12)
    dot = float(abs(np.dot(pred, gt)))
    dot = max(0.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def origin_error_normalized(
    pred_origin: np.ndarray,
    gt_origin: np.ndarray,
    gt_axis: np.ndarray,
    bbox_diag: float,
) -> float:
    axis = gt_axis.astype(np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    delta = pred_origin.astype(np.float64) - gt_origin.astype(np.float64)
    perp = delta - np.dot(delta, axis) * axis
    return float(np.linalg.norm(perp) / max(float(bbox_diag), 1e-12))


def score_row(
    gt_type: str,
    pred_type: str,
    pred_axis: np.ndarray,
    pred_origin: np.ndarray,
    gt_axis: np.ndarray,
    gt_origin: np.ndarray,
    bbox_diag: float,
) -> Dict[str, object]:
    axis_err = axis_error_deg(pred_axis, gt_axis)
    origin_err = origin_error_normalized(pred_origin, gt_origin, gt_axis, bbox_diag)
    type_ok = canonical_joint_type(pred_type) == canonical_joint_type(gt_type)
    return {
        "pred_joint_type": canonical_joint_type(pred_type) or pred_type,
        "joint_type_correct": type_ok,
        "axis_error_deg": axis_err,
        "origin_error_normalized": origin_err,
        "success_strict": type_ok and axis_err < 10.0 and origin_err < 0.05,
        "success_relaxed": type_ok and axis_err < 15.0 and origin_err < 0.10,
        "success_axis_only": type_ok and axis_err < 10.0,
    }


def aggregate_downstream(rows: List[Dict[str, object]], setting: str) -> Dict[str, object]:
    n = len(rows)
    axis = summarize(parse_float(row.get("axis_error_deg")) for row in rows)
    origin = summarize(parse_float(row.get("origin_error_normalized")) for row in rows)
    type_ok = sum(parse_bool(row.get("joint_type_correct")) for row in rows)
    return {
        "Setting": setting,
        "Type Acc.": fmt_ratio(type_ok, n),
        "Strict": fmt_ratio(sum(parse_bool(row.get("success_strict")) for row in rows), n),
        "Relaxed": fmt_ratio(sum(parse_bool(row.get("success_relaxed")) for row in rows), n),
        "Axis-only": fmt_ratio(sum(parse_bool(row.get("success_axis_only")) for row in rows), n),
        "Axis Err.": f"{fmt_float(axis['mean'], 2)} deg / {fmt_float(axis['median'], 2)} deg",
        "Origin Err.": f"{fmt_float(origin['mean'], 3)} / {fmt_float(origin['median'], 3)}",
    }


def aggregate_mask(metrics_rows: List[Dict[str, str]]) -> Dict[str, object]:
    ious = [parse_float(row.get("iou")) for row in metrics_rows]
    iou_summary = summarize(ious)
    success = sum((parse_float(row.get("iou")) or 0.0) >= 0.5 for row in metrics_rows)
    return {
        "miou": iou_summary["mean"],
        "median_iou": iou_summary["median"],
        "success_at_0p5": success,
        "n": len(metrics_rows),
    }


def load_mask_rows(path: Optional[Path]) -> Optional[List[Dict[str, str]]]:
    if path is None:
        return None
    if not path.exists():
        return None
    return read_csv(path)


def read_gpt_type_rows(path: Optional[Path], manifest_cases: Dict[str, Dict[str, object]]) -> Optional[List[Dict[str, object]]]:
    if path is None:
        return None
    rows = []
    for row in read_csv(path):
        case_id = str(row.get("case_id") or row.get("id") or "").strip()
        if not case_id:
            continue
        gt_type = canonical_joint_type(row.get("gt_joint_type") or row.get("gt_type") or manifest_cases[case_id]["gt_joint_type"])
        pred_key = next(
            (
                key
                for key in ["pred_joint_type", "pred_type", "gpt_joint_type", "joint_type", "response"]
                if str(row.get(key, "")).strip()
            ),
            None,
        )
        pred_type = canonical_joint_type(row.get(pred_key)) if pred_key else None
        rows.append(
            {
                "case_id": case_id,
                "gt_joint_type": gt_type,
                "pred_joint_type": pred_type or "unparsed",
                "joint_type_correct": pred_type == gt_type,
            }
        )
    return rows


def gpt_replace_type_downstream(
    provided_rows: List[Dict[str, str]],
    gpt_rows: Optional[List[Dict[str, object]]],
) -> Optional[List[Dict[str, object]]]:
    if gpt_rows is None:
        return None
    gpt_by_case = {str(row["case_id"]): row for row in gpt_rows}
    out = []
    for row in provided_rows:
        case_id = row["case_id"]
        gpt = gpt_by_case.get(case_id)
        if gpt is None:
            continue
        pred_type = str(gpt["pred_joint_type"])
        gt_type = str(row.get("gt_joint_type") or row.get("gt_type"))
        type_ok = canonical_joint_type(pred_type) == canonical_joint_type(gt_type)
        axis_err = parse_float(row["axis_error_deg"])
        origin_err = parse_float(row["origin_error_normalized"])
        out.append(
            {
                **row,
                "pred_joint_type": pred_type,
                "pred_type": pred_type,
                "joint_type_correct": type_ok,
                "success_strict": type_ok and (axis_err or 999.0) < 10.0 and (origin_err or 999.0) < 0.05,
                "success_relaxed": type_ok and (axis_err or 999.0) < 15.0 and (origin_err or 999.0) < 0.10,
                "success_axis_only": type_ok and (axis_err or 999.0) < 10.0,
            }
        )
    return out


def mask_gated_downstream(
    provided_rows: List[Dict[str, str]],
    p3sam_metrics_rows: List[Dict[str, str]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """Upper-bound propagation diagnostic.

    The locked KPP CSV was measured with provided masks. To avoid mixing in a
    separate GLB/export bridge, this diagnostic keeps the locked KPP axis/type
    predictions and only gates success by whether P3-SAM produced a usable
    binary mask. It answers: "even if KPP were unchanged, how many cases remain
    valid after the optional mask front-end?"
    """
    iou_by_case = {
        str(row["case_id"]): (parse_float(row.get("iou")) or 0.0)
        for row in p3sam_metrics_rows
    }
    out: List[Dict[str, object]] = []
    for row in provided_rows:
        iou = iou_by_case.get(str(row["case_id"]), 0.0)
        mask_ok = iou >= iou_threshold
        out.append(
            {
                **row,
                "joint_type_correct": parse_bool(row.get("joint_type_correct")),
                "success_strict": mask_ok and parse_bool(row.get("success_strict")),
                "success_relaxed": mask_ok and parse_bool(row.get("success_relaxed")),
                "success_axis_only": mask_ok and parse_bool(row.get("success_axis_only")),
                "mask_iou": iou,
                "mask_ok": mask_ok,
            }
        )
    return out


def localization_drop_row(
    provided_rows: List[Dict[str, str]],
    gated_rows: List[Dict[str, object]],
    label: str,
) -> Dict[str, object]:
    provided = aggregate_downstream(provided_rows, "provided")
    gated = aggregate_downstream(gated_rows, "gated")

    def drop_metric(key: str) -> str:
        p = provided[key]
        g = gated[key]
        return f"{p} -> {g}"

    return {
        "Setting": label,
        "Type Acc.": drop_metric("Type Acc."),
        "Strict": drop_metric("Strict"),
        "Relaxed": drop_metric("Relaxed"),
        "Axis-only": drop_metric("Axis-only"),
        "Axis Err.": drop_metric("Axis Err."),
        "Origin Err.": drop_metric("Origin Err."),
    }


def binary_face_from_p3sam(
    mesh: trimesh.Trimesh,
    pred_mask_path: Path,
    pred_mask_format: str,
    matched_part_id: Optional[int],
) -> np.ndarray:
    raw = np.asarray(load_mask_array(str(pred_mask_path))).reshape(-1)
    if pred_mask_format == "binary_face":
        if raw.shape[0] != len(mesh.faces):
            raise ValueError(f"binary_face length {raw.shape[0]} != mesh faces {len(mesh.faces)}")
        return raw.astype(bool)
    if pred_mask_format == "binary_vertex":
        if raw.shape[0] != len(mesh.vertices):
            raise ValueError(f"binary_vertex length {raw.shape[0]} != mesh vertices {len(mesh.vertices)}")
        return raw.astype(bool)[mesh.faces].mean(axis=1) > 0.5
    if pred_mask_format == "vertex_labels":
        if raw.shape[0] != len(mesh.vertices):
            raise ValueError(f"vertex_labels length {raw.shape[0]} != mesh vertices {len(mesh.vertices)}")
        labels = raw.astype(np.int64)
        if matched_part_id is None:
            raise ValueError("matched_part_id is required for vertex_labels")
        return (labels[mesh.faces] == int(matched_part_id)).mean(axis=1) > 0.5
    if pred_mask_format == "face_labels":
        if raw.shape[0] != len(mesh.faces):
            raise ValueError(f"face_labels length {raw.shape[0]} != mesh faces {len(mesh.faces)}")
        if matched_part_id is None:
            raise ValueError("matched_part_id is required for face_labels")
        return raw.astype(np.int64) == int(matched_part_id)
    raise ValueError(f"Unsupported pred_mask_format: {pred_mask_format}")


def run_p3sam_kpp(
    *,
    manifest_cases: Dict[str, Dict[str, object]],
    p3sam_manifest_csv: Path,
    p3sam_metrics_csv: Path,
    kpp_checkpoint: Path,
    output_csv: Path,
    device: torch.device,
    num_points: int,
    max_cases: Optional[int],
) -> List[Dict[str, object]]:
    import torch
    from dragmesh.inference.inference_animation_kpp import _prepare_kpp_inputs_from_mesh, load_kpp_model
    from dragmesh.utils.kpp_normalization import denormalize_points

    p3sam_manifest = {row["case_id"]: row for row in read_csv(p3sam_manifest_csv)}
    p3sam_metrics = {row["case_id"]: row for row in read_csv(p3sam_metrics_csv)}
    model = load_kpp_model(str(kpp_checkpoint), device)
    if model is None:
        raise RuntimeError(f"Failed to load KPP checkpoint: {kpp_checkpoint}")
    model.eval()

    rows: List[Dict[str, object]] = []
    case_ids = [case_id for case_id in manifest_cases if case_id in p3sam_manifest]
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    for idx, case_id in enumerate(case_ids, start=1):
        case = manifest_cases[case_id]
        src = p3sam_manifest[case_id]
        p3row = p3sam_metrics.get(case_id, {})
        mesh = load_trimesh(Path(src["mesh_path"]))
        matched_part = str(p3row.get("matched_pred_part_id", "")).strip()
        matched_part_id = int(matched_part) if matched_part else None
        face_mask = binary_face_from_p3sam(
            mesh=mesh,
            pred_mask_path=Path(src["pred_mask_path"]),
            pred_mask_format=src.get("pred_mask_format", "face_labels"),
            matched_part_id=matched_part_id,
        )
        vertex_labels = face_binary_to_vertex_labels(mesh.faces, len(mesh.vertices), face_mask)
        vertex_mask = vertex_labels > 0

        drag_point = np.asarray(case["drag_3d_start"], dtype=np.float32)
        drag_vector = np.asarray(case["drag_3d_vector"], dtype=np.float32)
        pc, sampled_mask, drag_point_norm, drag_vector_norm, center, scale = _prepare_kpp_inputs_from_mesh(
            initial_mesh=mesh,
            vertex_part_mask=vertex_mask.astype(np.int32),
            drag_point_world=drag_point,
            drag_vector_world=drag_vector,
            num_points=num_points,
        )

        with torch.no_grad():
            pred_type_logits, pred_axis, pred_origin = model(
                torch.from_numpy(pc).float().unsqueeze(0).to(device),
                torch.from_numpy(sampled_mask).float().unsqueeze(0).to(device),
                torch.from_numpy(drag_point_norm).float().unsqueeze(0).to(device),
                torch.from_numpy(drag_vector_norm).float().unsqueeze(0).to(device),
            )

        pred_axis_np = pred_axis.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_axis_np = pred_axis_np / (np.linalg.norm(pred_axis_np) + 1e-8)
        pred_origin_norm = pred_origin.squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_origin_world = denormalize_points(pred_origin_norm, center, scale)
        pred_type_idx = int(torch.argmax(pred_type_logits, dim=-1).item())
        pred_type = "revolute" if pred_type_idx == 0 else "prismatic"
        gt_type = str(case["gt_joint_type"])
        gt_axis = np.asarray(case["gt_axis"], dtype=np.float32)
        gt_origin = np.asarray(case["gt_origin"], dtype=np.float32)
        bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        scored = score_row(gt_type, pred_type, pred_axis_np, pred_origin_world, gt_axis, gt_origin, bbox_diag)
        rows.append(
            {
                "case_id": case_id,
                "category": case["category"],
                "gt_joint_type": canonical_joint_type(gt_type),
                "pred_joint_type": scored["pred_joint_type"],
                "joint_type_correct": scored["joint_type_correct"],
                "axis_error_deg": scored["axis_error_deg"],
                "origin_error_normalized": scored["origin_error_normalized"],
                "success_strict": scored["success_strict"],
                "success_relaxed": scored["success_relaxed"],
                "success_axis_only": scored["success_axis_only"],
                "mask_iou": parse_float(p3row.get("iou")),
                "mask_precision": parse_float(p3row.get("precision")),
                "mask_recall": parse_float(p3row.get("recall")),
                "matched_pred_part_id": "" if matched_part_id is None else matched_part_id,
                "pred_positive_faces": int(face_mask.sum()),
            }
        )
        print(f"[{idx:02d}/{len(case_ids):02d}] {case_id}: type={pred_type}, axis={scored['axis_error_deg']:.2f}, origin={scored['origin_error_normalized']:.3f}, mask_iou={fmt_float(parse_float(p3row.get('iou')), 3)}")

    write_csv(
        output_csv,
        rows,
        [
            "case_id",
            "category",
            "gt_joint_type",
            "pred_joint_type",
            "joint_type_correct",
            "axis_error_deg",
            "origin_error_normalized",
            "success_strict",
            "success_relaxed",
            "success_axis_only",
            "mask_iou",
            "mask_precision",
            "mask_recall",
            "matched_pred_part_id",
            "pred_positive_faces",
        ],
    )
    return rows


def table_to_markdown(rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> str:
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join("---" for _ in fieldnames) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |")
    return "\n".join(lines)


def table_to_latex(rows: List[Dict[str, object]], fieldnames: Sequence[str], caption: str, label: str) -> str:
    cols = "l" * len(fieldnames)
    body = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule"]
    body.append(" & ".join(fieldnames) + " \\\\")
    body.append("\\midrule")
    for row in rows:
        body.append(" & ".join(str(row.get(name, "")) for name in fieldnames) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_csv", type=Path, required=True)
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--p3sam_metrics_csv", type=Path, default=None, help="Fallback mask CSV for the downstream gate row.")
    parser.add_argument("--raw_p3sam_metrics_csv", type=Path, default=Path("results/p3sam_gap25_nonoracle_local_region_raw/per_case_metrics.csv"))
    parser.add_argument("--fixed_p3sam_metrics_csv", type=Path, default=Path("results/p3sam_gap25_nonoracle_local_union_fixed/per_case_metrics.csv"))
    parser.add_argument("--category_p3sam_metrics_csv", type=Path, default=Path("results/p3sam_gap25_nonoracle_local_union_loco/per_case_metrics.csv"))
    parser.add_argument("--oracle_p3sam_metrics_csv", type=Path, default=Path("results/p3sam_gap25_oracle_centroid_upper_bound/per_case_metrics.csv"))
    parser.add_argument("--p3sam_manifest_csv", type=Path, required=True)
    parser.add_argument("--gpt_type_csv", type=Path, default=None, help="Optional CSV with case_id and pred_joint_type/gpt_joint_type/response.")
    parser.add_argument("--kpp_checkpoint", type=Path, default=Path("outputs/kpp_full_bbox_center_full_gapartnet_kpp_20260503_120142/best_model_kpp.pth"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/reviewer_q4_diagnostics_nonoracle"))
    parser.add_argument("--run_p3sam_kpp", action="store_true", help="Rerun KPP with saved P3-SAM masks.")
    parser.add_argument("--p3sam_kpp_csv", type=Path, default=None, help="Reuse an existing P3-SAM+KPP CSV instead of rerunning.")
    parser.add_argument("--include_p3sam_kpp_downstream", action="store_true",
                        help="Include the experimental P3-SAM-mask KPP rerun row. Off by default because it requires a matched mesh/loader bridge.")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    provided_rows = read_csv(args.trajectory_csv)
    manifest_cases = load_manifest_cases(args.manifest_json)
    raw_mask_rows = load_mask_rows(args.raw_p3sam_metrics_csv)
    fixed_mask_rows = load_mask_rows(args.fixed_p3sam_metrics_csv)
    category_mask_rows = load_mask_rows(args.category_p3sam_metrics_csv)
    oracle_mask_rows = load_mask_rows(args.oracle_p3sam_metrics_csv)
    fallback_mask_rows = read_csv(args.p3sam_metrics_csv) if args.p3sam_metrics_csv is not None else category_mask_rows
    mask_summary = aggregate_mask(fallback_mask_rows) if fallback_mask_rows is not None else {"miou": None, "median_iou": None, "success_at_0p5": 0, "n": 0}
    gpt_rows = read_gpt_type_rows(args.gpt_type_csv, manifest_cases)

    p3sam_kpp_rows: Optional[List[Dict[str, object]]] = None
    p3sam_kpp_csv = args.p3sam_kpp_csv or (args.output_dir / "p3sam_mask_kpp_type_downstream.csv")
    if args.p3sam_kpp_csv is not None and args.p3sam_kpp_csv.exists():
        p3sam_kpp_rows = read_csv(args.p3sam_kpp_csv)
    elif args.run_p3sam_kpp:
        if args.device == "cuda":
            device = torch.device("cuda")
        elif args.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p3sam_kpp_rows = run_p3sam_kpp(
            manifest_cases=manifest_cases,
            p3sam_manifest_csv=args.p3sam_manifest_csv,
            p3sam_metrics_csv=args.p3sam_metrics_csv,
            kpp_checkpoint=args.kpp_checkpoint,
            output_csv=p3sam_kpp_csv,
            device=device,
            num_points=args.num_points,
            max_cases=args.max_cases,
        )

    gpt_downstream_rows = gpt_replace_type_downstream(provided_rows, gpt_rows)
    def mask_row(label: str, rows: Optional[List[Dict[str, str]]]) -> Optional[Dict[str, object]]:
        if rows is None:
            return None
        row = {
            "Component": label,
            "Input": "mesh + user drag + P3-SAM proposals" if "manifest_drag" in str(rows[0].get("prompt_mode", "")) else "mesh + drag + P3-SAM proposals",
            "Metric": "mIoU / Success@0.5",
            "Result": f"{fmt_float(aggregate_mask(rows)['miou'], 3)} / {fmt_ratio(int(aggregate_mask(rows)['success_at_0p5']), int(aggregate_mask(rows)['n']))}",
            "Failure mode": "over-segmentation / missing movable part after cleanup",
        }
        return row

    table_a: List[Dict[str, object]] = []
    if raw_mask_rows is not None:
        table_a.append({
            "Component": "Raw non-oracle P3-SAM",
            "Input": "user drag + proposal selection",
            "Metric": "mIoU / Success@0.5",
            "Result": f"{fmt_float(aggregate_mask(raw_mask_rows)['miou'], 3)} / {fmt_ratio(int(aggregate_mask(raw_mask_rows)['success_at_0p5']), int(aggregate_mask(raw_mask_rows)['n']))}",
            "Failure mode": "proposal mismatch / coordinate bridge failure",
        })
    if fixed_mask_rows is not None:
        table_a.append({
            "Component": "Fixed drag-aware cleanup",
            "Input": "user drag + proposal selection",
            "Metric": "mIoU / Success@0.5",
            "Result": f"{fmt_float(aggregate_mask(fixed_mask_rows)['miou'], 3)} / {fmt_ratio(int(aggregate_mask(fixed_mask_rows)['success_at_0p5']), int(aggregate_mask(fixed_mask_rows)['n']))}",
            "Failure mode": "over-segmentation / missing movable part after cleanup",
        })
    if category_mask_rows is not None:
        table_a.append({
            "Component": "Category-adaptive drag-aware cleanup",
            "Input": "user drag + proposal selection",
            "Metric": "mIoU / Success@0.5",
            "Result": f"{fmt_float(aggregate_mask(category_mask_rows)['miou'], 3)} / {fmt_ratio(int(aggregate_mask(category_mask_rows)['success_at_0p5']), int(aggregate_mask(category_mask_rows)['n']))}",
            "Failure mode": "category mismatch / insufficient cleanup",
        })
    if oracle_mask_rows is not None:
        table_a.append({
            "Component": "Oracle centroid upper bound",
            "Input": "GT movable-part centroid",
            "Metric": "mIoU / Success@0.5",
            "Result": f"{fmt_float(aggregate_mask(oracle_mask_rows)['miou'], 3)} / {fmt_ratio(int(aggregate_mask(oracle_mask_rows)['success_at_0p5']), int(aggregate_mask(oracle_mask_rows)['n']))}",
            "Failure mode": "oracle prompt; not annotation-free",
        })

    table_a.insert(
        0,
        {
            "Component": "KPP joint-type head",
            "Input": "mesh + drag",
            "Metric": "Type Acc.",
            "Result": aggregate_downstream(provided_rows, "tmp")["Type Acc."],
            "Failure mode": "revolute/prismatic confusion",
        },
    )
    table_a.insert(
        1,
        {
            "Component": "GPT-4o joint-type prior",
            "Input": "rendered view + prompt",
            "Metric": "Type Acc.",
            "Result": aggregate_downstream(gpt_rows, "tmp")["Type Acc."] if gpt_rows else "N/A (no audit log supplied)",
            "Failure mode": "ambiguous affordance / occlusion",
        },
    )
    table_a_fields = ["Component", "Input", "Metric", "Result", "Failure mode"]

    table_b = [aggregate_downstream(provided_rows, "Provided mask + KPP type")]
    if raw_mask_rows is not None:
        gated_raw = mask_gated_downstream(provided_rows, raw_mask_rows, iou_threshold=0.5)
        table_b.append(aggregate_downstream(gated_raw, "raw_drag_selected P3-SAM front end + KPP motion"))
    if fixed_mask_rows is not None:
        gated_fixed = mask_gated_downstream(provided_rows, fixed_mask_rows, iou_threshold=0.5)
        table_b.append(aggregate_downstream(gated_fixed, "fixed_drag_cleanup P3-SAM front end + KPP motion"))
    if category_mask_rows is not None:
        gated_category = mask_gated_downstream(provided_rows, category_mask_rows, iou_threshold=0.5)
        table_b.append(aggregate_downstream(gated_category, "category_adaptive_drag_cleanup P3-SAM front end + KPP motion"))
        table_b.append(localization_drop_row(provided_rows, gated_category, "Drop due to P3-SAM localization"))
    else:
        gate_rows = fallback_mask_rows or []
        if gate_rows:
            table_b.append(
                aggregate_downstream(
                    mask_gated_downstream(provided_rows, gate_rows, iou_threshold=0.5),
                    "P3-SAM gate + provided-mask KPP upper bound",
                )
            )
    if p3sam_kpp_rows is not None and args.include_p3sam_kpp_downstream:
        table_b.append(aggregate_downstream(p3sam_kpp_rows, "P3-SAM mask + KPP type"))
    if gpt_downstream_rows is None:
        table_b.append(
            {
                "Setting": "Provided mask + GPT-4o type",
                "Type Acc.": "N/A",
                "Strict": "N/A",
                "Relaxed": "N/A",
                "Axis-only": "N/A",
                "Axis Err.": "N/A",
                "Origin Err.": "N/A",
            }
        )
    else:
        table_b.append(aggregate_downstream(gpt_downstream_rows, "Provided mask + GPT-4o type"))
    table_b_fields = ["Setting", "Type Acc.", "Strict", "Relaxed", "Axis-only", "Axis Err.", "Origin Err."]

    write_csv(args.output_dir / "table_a_optional_foundation_diagnostics.csv", table_a, table_a_fields)
    write_csv(args.output_dir / "table_b_downstream_effect.csv", table_b, table_b_fields)
    (args.output_dir / "table_a_optional_foundation_diagnostics.tex").write_text(
        table_to_latex(table_a, table_a_fields, "Optional foundation-model diagnostics on GAP-25.", "tab:optional_foundation_diagnostics")
    )
    (args.output_dir / "table_b_downstream_effect.tex").write_text(
        table_to_latex(table_b, table_b_fields, "Downstream effect of automatic components on GAP-25.", "tab:automatic_component_downstream")
    )
    report_lines = [
        "# Reviewer Q4 Diagnostic Tables",
        "",
        "## Table A: Optional foundation-model diagnostics on GAP-25",
        "",
        table_to_markdown(table_a, table_a_fields),
        "",
        "## Table B: Downstream effect of automatic components",
        "",
        table_to_markdown(table_b, table_b_fields),
        "",
        "Note: GPT-4o rows are only filled when `--gpt_type_csv` is supplied. The P3-SAM rows are localization-gated diagnostics: KPP predictions are kept fixed to the provided-mask setting to isolate the effect of target-part localization.",
    ]
    (args.output_dir / "reviewer_q4_diagnostic_tables.md").write_text("\n".join(report_lines))
    print(f"Wrote {args.output_dir / 'reviewer_q4_diagnostic_tables.md'}")
    print(table_to_markdown(table_a, table_a_fields))
    print()
    print(table_to_markdown(table_b, table_b_fields))


if __name__ == "__main__":
    main()
