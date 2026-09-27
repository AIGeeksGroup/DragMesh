#!/usr/bin/env python3
"""Failure diagnostics for GAP-25 P3-SAM drag masks.

The proposal-oracle columns are analysis-only. They are never used by the
non-oracle front-end; they only explain whether the right P3-SAM proposal was
available and where it ranked under the drag-local score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.p3sam.postprocess_p3sam_drag_masks_nonoracle import category_from_case_id


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_by_case(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row["case_id"]: row for row in read_csv(path)}


def resolve_existing(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if not path.is_absolute() and (root / path).exists():
        return root / path
    name = path.name
    case_dir = path.parent.name
    local = root / "results" / "p3sam_gap25_assets" / case_dir / name
    if local.exists():
        return local
    return path


def classify_failure(row: Dict[str, object]) -> str:
    iou = float(row.get("selected_proposal_iou") or row.get("fixed_iou") or row.get("raw_iou") or 0.0)
    best_iou = float(row.get("best_proposal_iou") or 0.0)
    pred_area = float(row.get("pred_area_ratio") or 0.0)
    local_overlap = float(row.get("selected_local_overlap") or 0.0)
    drag_dist = float(row.get("drag_to_gt_movable_dist_norm") or 0.0)
    if drag_dist > 0.35:
        return "drag point mapped to wrong surface"
    if best_iou >= 0.5 and iou < 0.5:
        return "selected static/background region"
    if iou < 0.15 and pred_area < 0.08:
        return "mask too small"
    if iou < 0.15 and pred_area > 0.50:
        return "selected far-side region"
    if iou < 0.30 and local_overlap < 0.05:
        return "selected far-side region"
    if iou < 0.30:
        return "mixed/weak mask"
    return "mixed/weak mask"


def proposal_diagnostics(
    src: Dict[str, str],
    selected_row: Optional[Dict[str, str]],
    root: Path,
) -> Dict[str, object]:
    from scripts.evaluation.evaluate_p3sam_mask_metrics import as_face_array, binary_metrics, load_mask
    from scripts.p3sam.postprocess_p3sam_drag_masks_nonoracle import load_mesh_fast
    from scripts.p3sam.run_p3sam_local_union_pipeline import FIXED_PARAMS, local_seed_mask, score_proposals

    mesh_path = resolve_existing(src["mesh_path"], root)
    pred_path = resolve_existing(src["pred_mask_path"], root)
    gt_path = resolve_existing(src["gt_mask_path"], root)
    mesh = load_mesh_fast(mesh_path)
    pred_face = as_face_array(load_mask(pred_path), src.get("pred_mask_format", "face_labels"), mesh).astype(np.int64)
    gt_face = as_face_array(load_mask(gt_path), src.get("gt_mask_format", "binary_face"), mesh).astype(bool)

    labels = [int(v) for v in np.unique(pred_face) if int(v) >= 0]
    scored = []
    for label in labels:
        mask = pred_face == label
        precision, recall, iou = binary_metrics(mask, gt_face)
        scored.append((float(iou), int(label), float(precision), float(recall), float(mask.mean())))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_label, best_precision, best_recall, best_area = scored[0] if scored else (0.0, -1, 0.0, 0.0, 0.0)

    selected_label = -1
    selected_iou = 0.0
    selected_local_overlap = 0.0
    rank = 999
    if selected_row is not None:
        labels_text = selected_row.get("selected_labels") or "[]"
        try:
            parsed = json.loads(labels_text)
            if parsed:
                selected_label = int(parsed[0])
        except json.JSONDecodeError:
            selected_label = int(selected_row.get("selected_label", -1) or -1)
        for iou, label, _, _, _ in scored:
            if label == selected_label:
                selected_iou = float(iou)
                break
        prompt_keys = ["mapped_surface_x", "mapped_surface_y", "mapped_surface_z"]
        if all(k in selected_row and selected_row[k] != "" for k in prompt_keys):
            prompt = np.asarray([float(selected_row[k]) for k in prompt_keys], dtype=np.float64)
            drag_vec = np.zeros(3, dtype=np.float64)
            local_mask = local_seed_mask(mesh, prompt, drag_vec, FIXED_PARAMS)
            candidates = score_proposals(mesh, pred_face, prompt, drag_vec, local_mask)
            ordered = [int(c["label"]) for c in candidates]
            if best_label in ordered:
                rank = ordered.index(best_label) + 1
            for cand in candidates:
                if int(cand["label"]) == selected_label:
                    selected_local_overlap = float(cand["overlap_with_local"])
                    break

    return {
        "best_proposal_iou": best_iou,
        "best_proposal_label": best_label,
        "best_proposal_rank": rank,
        "best_proposal_precision": best_precision,
        "best_proposal_recall": best_recall,
        "best_proposal_area_ratio": best_area,
        "selected_proposal_label": selected_label,
        "selected_proposal_iou": selected_iou,
        "selected_local_overlap": selected_local_overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3sam_manifest", type=Path, default=None, help="Original P3-SAM proposal manifest. Optional.")
    parser.add_argument("--raw_metrics_csv", type=Path, default=Path("results/p3sam_gap25_nonoracle_local_region_raw/per_case_metrics.csv"))
    parser.add_argument("--fixed_metrics_csv", type=Path, default=Path("results/p3sam_gap25_nonoracle_local_union_fixed/per_case_metrics.csv"))
    parser.add_argument("--category_metrics_csv", type=Path, default=Path("results/p3sam_gap25_nonoracle_local_union_loco/per_case_metrics.csv"))
    parser.add_argument("--oracle_metrics_csv", type=Path, default=Path("results/p3sam_gap25_oracle_centroid_upper_bound/per_case_metrics.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/p3sam_gap25_debug_failure"))
    args = parser.parse_args()

    raw = load_by_case(args.raw_metrics_csv)
    fixed = load_by_case(args.fixed_metrics_csv)
    category = load_by_case(args.category_metrics_csv)
    oracle = load_by_case(args.oracle_metrics_csv)
    case_ids = sorted(set(raw) | set(fixed) | set(category) | set(oracle))

    proposal_rows = read_csv(args.p3sam_manifest) if args.p3sam_manifest and args.p3sam_manifest.exists() else []
    proposals = {row["case_id"]: row for row in proposal_rows}

    rows: List[Dict[str, object]] = []
    for case_id in case_ids:
        fixed_row = fixed.get(case_id) or raw.get(case_id) or {}
        category_name = fixed_row.get("category") or raw.get(case_id, {}).get("category") or category_from_case_id(case_id)
        row: Dict[str, object] = {
            "case_id": case_id,
            "category": category_name,
            "raw_iou": raw.get(case_id, {}).get("iou", ""),
            "fixed_iou": fixed.get(case_id, {}).get("iou", ""),
            "category_iou": category.get(case_id, {}).get("iou", ""),
            "oracle_iou": oracle.get(case_id, {}).get("iou", ""),
            "drag_on_gt_movable": fixed_row.get("mapped_surface_on_gt", ""),
            "drag_to_gt_movable_dist_norm": fixed_row.get("mapped_to_gt_movable_dist_norm", ""),
            "pred_area_ratio": fixed_row.get("pred_area_ratio", ""),
        }
        if case_id in proposals:
            row.update(proposal_diagnostics(proposals[case_id], fixed_row, REPO_ROOT))
        else:
            row.update(
                {
                    "best_proposal_iou": "",
                    "best_proposal_label": "",
                    "best_proposal_rank": "",
                    "best_proposal_precision": "",
                    "best_proposal_recall": "",
                    "best_proposal_area_ratio": "",
                    "selected_proposal_label": fixed_row.get("selected_labels", ""),
                    "selected_proposal_iou": fixed_row.get("iou", ""),
                    "selected_local_overlap": fixed_row.get("top_candidate_overlap", ""),
                }
            )
        row["failure_type"] = classify_failure(row)
        rows.append(row)

    has_proposal_manifest = bool(proposals)
    for k in [1, 2, 3, 5, 8]:
        if has_proposal_manifest:
            count = sum(
                str(row.get("best_proposal_rank", "")).isdigit()
                and int(row["best_proposal_rank"]) <= k
                and float(row.get("best_proposal_iou") or 0.0) >= 0.5
                for row in rows
            )
            print(f"top{k}_contains_correct_proposal={count}/{len(rows)}")
        else:
            print(f"top{k}_contains_correct_proposal=unavailable (no P3-SAM proposal manifest)")

    fields = [
        "case_id",
        "category",
        "raw_iou",
        "fixed_iou",
        "category_iou",
        "oracle_iou",
        "drag_on_gt_movable",
        "drag_to_gt_movable_dist_norm",
        "best_proposal_iou",
        "best_proposal_label",
        "best_proposal_rank",
        "selected_proposal_label",
        "selected_proposal_iou",
        "selected_local_overlap",
        "best_proposal_precision",
        "best_proposal_recall",
        "best_proposal_area_ratio",
        "pred_area_ratio",
        "failure_type",
    ]
    write_csv(args.output_dir / "failure_diagnosis.csv", rows, fields)

    summary: Dict[str, object] = {}
    for cat in sorted({str(row["category"]) for row in rows}):
        subset = [row for row in rows if row["category"] == cat]
        hist = Counter(str(row["failure_type"]) for row in subset)
        summary[cat] = {"n": len(subset), **dict(sorted(hist.items()))}
    if has_proposal_manifest:
        summary["topk_correct_proposal_counts"] = {
            str(k): sum(
                str(row.get("best_proposal_rank", "")).isdigit()
                and int(row["best_proposal_rank"]) <= k
                and float(row.get("best_proposal_iou") or 0.0) >= 0.5
                for row in rows
            )
            for k in [1, 2, 3, 5, 8]
        }
    else:
        summary["topk_correct_proposal_counts"] = "unavailable_no_p3sam_proposal_manifest"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "category_failure_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output_dir / 'failure_diagnosis.csv'}")
    print(f"Wrote {args.output_dir / 'category_failure_summary.json'}")


if __name__ == "__main__":
    main()
