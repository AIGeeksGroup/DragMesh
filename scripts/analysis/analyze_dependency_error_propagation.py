#!/usr/bin/env python3
"""Audit upstream dependency accuracy and downstream error propagation.

Reports accuracy, failure modes, and downstream propagation for optional
joint-type classifiers (GPT/VLM or KPP head) and segmentation front ends such
as P3-SAM. This script intentionally works from locked eval
artifacts: it does not rerun inference or alter metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional


BOOL_TRUE = {"true", "1", "yes", "y", "t"}


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in BOOL_TRUE


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    return float(text)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def fmt_ratio(numer: int, denom: int) -> str:
    pct = 100.0 * numer / denom if denom else 0.0
    return f"{numer}/{denom} ({pct:.1f}%)"


def summarize(values: Iterable[Optional[float]]) -> Dict[str, object]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": "N/A", "median": "N/A", "max": "N/A"}
    vals_sorted = sorted(vals)
    return {
        "n": len(vals_sorted),
        "mean": mean(vals_sorted),
        "median": median(vals_sorted),
        "max": max(vals_sorted),
    }


def fmt_stat(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def first_present(row: Dict[str, object], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return key
    return None


def parse_segmentation_rows(segmentation_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    parsed = []
    for row in segmentation_rows:
        iou_key = first_present(row, ["iou", "IoU", "mask_iou", "p3sam_iou"])
        precision_key = first_present(row, ["precision", "Precision", "mask_precision"])
        recall_key = first_present(row, ["recall", "Recall", "mask_recall"])
        parsed.append(
            {
                "case_id": str(row.get("case_id", "unknown")),
                "iou": parse_float(row.get(iou_key)) if iou_key else None,
                "precision": parse_float(row.get(precision_key)) if precision_key else None,
                "recall": parse_float(row.get(recall_key)) if recall_key else None,
                "selection_mode": row.get("selection_mode", ""),
                "prompt_mode": row.get("prompt_mode", ""),
                "choose_head": row.get("choose_head", ""),
            }
        )
    return parsed


def load_optional_segmentation(path: Optional[Path]) -> Optional[List[Dict[str, str]]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Segmentation metrics file not found: {path}")
    return read_csv(path)


def load_optional_jsonl(path: Optional[Path]) -> Optional[List[Dict[str, object]]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"JSONL audit file not found: {path}")
    rows = []
    with path.open() as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def make_markdown_report(
    rows: List[Dict[str, object]],
    segmentation_rows: Optional[List[Dict[str, str]]],
    llm_audit_rows: Optional[List[Dict[str, object]]],
    output_csv: Path,
) -> str:
    n = len(rows)
    type_correct = sum(bool(row["type_correct"]) for row in rows)
    strict = sum(bool(row["success_strict"]) for row in rows)
    relaxed = sum(bool(row["success_relaxed"]) for row in rows)
    axis_only = sum(bool(row["success_axis_only"]) for row in rows)

    type_ok_rows = [row for row in rows if row["type_correct"]]
    type_bad_rows = [row for row in rows if not row["type_correct"]]
    axis_outliers = [row for row in rows if (row["axis_error_deg"] or 0.0) >= 45.0]
    origin_failures = [row for row in rows if (row["origin_error_normalized"] or 0.0) >= 0.05]

    confusion = Counter((row["gt_joint_type"], row["pred_joint_type"]) for row in rows)
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    scope_lines = [
        "# Dependency Error Propagation Audit",
        "",
        "## Scope",
        "",
        "- Source metrics: locked GAP-25 DragMesh/KPP evaluation CSV.",
        "- This audits the automatic KPP joint-type head used in the reported KPP-only setting.",
        "- GPT/VLM joint classification is optional in `inference_pipeline.py`; no GPT response log was found in the supplied DragMesh artifacts, so GPT-4o accuracy is not inferred here.",
    ]
    if segmentation_rows is None:
        scope_lines.append(
            "- P3-SAM is optional. No segmentation metrics CSV was supplied, so this report documents mask dependency and propagation risk instead of fabricating P3-SAM accuracy."
        )
    else:
        scope_lines.append(
            "- P3-SAM/mask metrics were supplied separately and are treated as an upstream-front-end audit; the locked kinematic CSV itself was measured with provided movable masks."
        )
    scope_lines += [
        "",
        "## Overall Accuracy",
        "",
        f"- Cases: `{n}`",
        f"- Joint type accuracy: `{fmt_ratio(type_correct, n)}`",
        f"- Strict / relaxed / axis-only success: `{fmt_ratio(strict, n)}` / `{fmt_ratio(relaxed, n)}` / `{fmt_ratio(axis_only, n)}`",
        f"- Axis error deg mean / median / max: `{fmt_stat(summarize(row['axis_error_deg'] for row in rows)['mean'])}` / `{fmt_stat(summarize(row['axis_error_deg'] for row in rows)['median'])}` / `{fmt_stat(summarize(row['axis_error_deg'] for row in rows)['max'])}`",
        f"- Origin error mean / median / max: `{fmt_stat(summarize(row['origin_error_normalized'] for row in rows)['mean'])}` / `{fmt_stat(summarize(row['origin_error_normalized'] for row in rows)['median'])}` / `{fmt_stat(summarize(row['origin_error_normalized'] for row in rows)['max'])}`",
        "",
        "## Joint-Type Confusion",
        "",
        "| GT | Pred | Count |",
        "| --- | --- | --- |",
    ]
    lines = scope_lines
    for (gt, pred), count in sorted(confusion.items()):
        lines.append(f"| {gt} | {pred} | {count} |")

    lines += [
        "",
        "## Downstream Propagation",
        "",
        "| Subset | Count | Strict | Relaxed | Axis-only | Mean final CD | Median final CD | Mean endpoint L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, subset in [("type correct", type_ok_rows), ("type wrong", type_bad_rows)]:
        denom = len(subset)
        final_cd = summarize(row["normalized_traj_cd_final"] for row in subset)
        endpoint = summarize(row["endpoint_l2_mean"] for row in subset)
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(denom),
                    fmt_ratio(sum(bool(row["success_strict"]) for row in subset), denom),
                    fmt_ratio(sum(bool(row["success_relaxed"]) for row in subset), denom),
                    fmt_ratio(sum(bool(row["success_axis_only"]) for row in subset), denom),
                    fmt_stat(final_cd["mean"]),
                    fmt_stat(final_cd["median"]),
                    fmt_stat(endpoint["mean"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Category Breakdown",
        "",
        "| Category | Type Acc | Strict | Relaxed | Axis-only | Mean Axis Deg | Mean Origin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in sorted(by_category):
        subset = by_category[category]
        denom = len(subset)
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    fmt_ratio(sum(bool(row["type_correct"]) for row in subset), denom),
                    fmt_ratio(sum(bool(row["success_strict"]) for row in subset), denom),
                    fmt_ratio(sum(bool(row["success_relaxed"]) for row in subset), denom),
                    fmt_ratio(sum(bool(row["success_axis_only"]) for row in subset), denom),
                    fmt_stat(summarize(row["axis_error_deg"] for row in subset)["mean"]),
                    fmt_stat(summarize(row["origin_error_normalized"] for row in subset)["mean"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Failure Modes",
        "",
        f"- Type errors: `{len(type_bad_rows)}/{n}`; cases: `{', '.join(row['case_id'] for row in type_bad_rows)}`.",
        f"- Axis outliers >=45 deg: `{len(axis_outliers)}/{n}`; cases: `{', '.join(row['case_id'] for row in axis_outliers)}`.",
        f"- Origin errors >=0.05 bbox diag: `{len(origin_failures)}/{n}`; cases: `{', '.join(row['case_id'] for row in origin_failures)}`.",
        "- Type mistakes switch the revolute/prismatic branch, so they directly zero out strict, relaxed, and axis-only success under the reported definitions and often raise trajectory CD.",
        "- Origin errors mostly explain the gap between axis-only success and strict success: many cases have a plausible axis but fail the tight origin threshold.",
        "",
        "## GPT/VLM Audit Log Status",
        "",
    ]

    if llm_audit_rows is None:
        lines += [
            "- No `inference_pipeline.py --audit_log` JSONL was supplied.",
            "- GPT/VLM accuracy is therefore not estimated from missing logs; the report uses the local KPP joint-type head as the measured automatic classifier.",
        ]
    else:
        source_counts = Counter(str(row.get("joint_type_source", "unknown")) for row in llm_audit_rows)
        llm_enabled = sum(bool(row.get("llm_enabled")) for row in llm_audit_rows)
        llm_used = sum(str(row.get("joint_type_source")) == "llm" for row in llm_audit_rows)
        source_summary = ", ".join(f"`{key}`={value}" for key, value in sorted(source_counts.items()))
        lines += [
            f"- Audit rows: `{len(llm_audit_rows)}`",
            f"- LLM enabled / used as final source: `{fmt_ratio(llm_enabled, len(llm_audit_rows))}` / `{fmt_ratio(llm_used, len(llm_audit_rows))}`",
            f"- Joint-type source counts: {source_summary}",
        ]

    lines += [
        "",
        "## Segmentation / P3-SAM Status",
        "",
    ]

    if segmentation_rows is None:
        lines += [
            "- No P3-SAM prediction metrics were supplied or found for this DragMesh run.",
            "- Therefore the current quantitative GAP-25 result should be described as using provided/ground-truth part labels for the movable mask.",
            "- P3-SAM is an optional annotation-free front end; report mask-error propagation qualitatively unless a P3-SAM-vs-GT mask CSV is generated.",
        ]
    else:
        parsed_seg = parse_segmentation_rows(segmentation_rows)
        seg_by_case = {row["case_id"]: row for row in parsed_seg}
        ious = [row["iou"] for row in parsed_seg]
        precisions = [row["precision"] for row in parsed_seg]
        recalls = [row["recall"] for row in parsed_seg]
        low_iou_cases = [str(row["case_id"]) for row in parsed_seg if row["iou"] is not None and row["iou"] < 0.25]
        high_iou_cases = [str(row["case_id"]) for row in parsed_seg if row["iou"] is not None and row["iou"] >= 0.50]
        iou_summary = summarize(ious)
        precision_summary = summarize(precisions)
        recall_summary = summarize(recalls)
        protocol_bits = []
        selection_modes = sorted({str(row["selection_mode"]) for row in parsed_seg if row["selection_mode"]})
        prompt_modes = sorted({str(row["prompt_mode"]) for row in parsed_seg if row["prompt_mode"]})
        choose_heads = sorted({str(row["choose_head"]) for row in parsed_seg if row["choose_head"]})
        if selection_modes:
            protocol_bits.append("selection=" + "/".join(selection_modes))
        if prompt_modes:
            protocol_bits.append("prompt=" + "/".join(prompt_modes))
        if choose_heads:
            protocol_bits.append("head=" + "/".join(choose_heads))
        protocol_text = "; ".join(protocol_bits) if protocol_bits else "mask CSV"

        lines += [
            f"- Segmentation rows: `{len(segmentation_rows)}`",
            f"- Protocol tag: `{protocol_text}`",
            f"- P3-SAM/mask IoU mean / median / max: `{fmt_stat(iou_summary['mean'])}` / `{fmt_stat(iou_summary['median'])}` / `{fmt_stat(iou_summary['max'])}`",
            f"- Precision mean / median: `{fmt_stat(precision_summary['mean'])}` / `{fmt_stat(precision_summary['median'])}`; recall mean / median: `{fmt_stat(recall_summary['mean'])}` / `{fmt_stat(recall_summary['median'])}`",
            f"- Low-IoU cases (<0.25): `{len(low_iou_cases)}/{len(segmentation_rows)}`; cases: `{', '.join(low_iou_cases)}`.",
            f"- Higher-IoU cases (>=0.50): `{len(high_iou_cases)}/{len(segmentation_rows)}`; cases: `{', '.join(high_iou_cases)}`.",
            "- These mask metrics are not merged into the locked KPP-only kinematic score, because that score was measured with provided masks. If the P3-SAM mask were substituted, low-IoU cases should be counted as upstream segmentation failures before KPP/DQ-VAE execution.",
            "",
            "### Mask-Conditioned Risk View",
            "",
            "| Mask subset | Count | Type acc under provided-mask KPP | Strict under provided-mask KPP | Mean mask IoU | Mean final CD |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]

        joined_rows = [row for row in rows if row["case_id"] in seg_by_case]
        subsets = [
            ("mask IoU < 0.25", [row for row in joined_rows if (seg_by_case[row["case_id"]]["iou"] or 0.0) < 0.25]),
            ("mask IoU >= 0.25", [row for row in joined_rows if (seg_by_case[row["case_id"]]["iou"] or 0.0) >= 0.25]),
            ("mask IoU >= 0.50", [row for row in joined_rows if (seg_by_case[row["case_id"]]["iou"] or 0.0) >= 0.50]),
        ]
        for name, subset in subsets:
            denom = len(subset)
            mask_stats = summarize(seg_by_case[row["case_id"]]["iou"] for row in subset)
            final_cd = summarize(row["normalized_traj_cd_final"] for row in subset)
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(denom),
                        fmt_ratio(sum(bool(row["type_correct"]) for row in subset), denom),
                        fmt_ratio(sum(bool(row["success_strict"]) for row in subset), denom),
                        fmt_stat(mask_stats["mean"]),
                        fmt_stat(final_cd["mean"]),
                    ]
                )
                + " |"
            )

    if segmentation_rows is None:
        p3sam_sentence = (
            "The current locked evaluation does not contain P3-SAM predictions; it uses provided part labels, "
            "so P3-SAM should be described as an optional front end whose mask errors would propagate into KPP point features and Gaussian binding rather than as an evaluated oracle."
        )
    else:
        parsed_seg = parse_segmentation_rows(segmentation_rows)
        iou_summary = summarize(row["iou"] for row in parsed_seg)
        low_iou = sum((row["iou"] or 0.0) < 0.25 for row in parsed_seg)
        p3sam_sentence = (
            "A separate P3-SAM/mask audit on the same GAP-25 cases gives mean/median IoU "
            f"{fmt_stat(iou_summary['mean'])}/{fmt_stat(iou_summary['median'])}, with {low_iou}/{len(parsed_seg)} cases below 0.25 IoU. "
            "Because the locked kinematic score uses provided masks, these P3-SAM numbers are reported as upstream-front-end accuracy rather than folded into the KPP-only trajectory score; substituting such masks would corrupt KPP point features and Gaussian binding before motion decoding."
        )

    lines += [
        "",
        "## Manuscript-Ready Paragraph",
        "",
        "For the automatic GAP-25 setting, DragMesh uses the local KPP joint-type head rather than a required GPT/VLM call. The head is correct on "
        f"{type_correct}/{n} cases. Type errors are not hidden by oracle labels: they switch the revolute/prismatic motion branch and reduce downstream success to "
        f"{strict}/{n} strict, {relaxed}/{n} relaxed, and {axis_only}/{n} axis-only successes. The failure tail is concentrated in ambiguous sliding-vs-hinge geometries and in origin localization: "
        f"{len(axis_outliers)}/{n} cases have axis error above 45 degrees, while {len(origin_failures)}/{n} exceed the 0.05 bbox-diagonal origin threshold. "
        + p3sam_sentence,
        "",
        f"Per-case CSV: `{output_csv}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory_csv",
        type=Path,
        required=True,
        help="Locked DragMesh trajectory/kinematic metrics CSV.",
    )
    parser.add_argument(
        "--segmentation_csv",
        type=Path,
        default=None,
        help="Optional P3-SAM or mask-vs-GT metrics CSV with a case_id column.",
    )
    parser.add_argument(
        "--llm_audit_jsonl",
        type=Path,
        default=None,
        help="Optional JSONL produced by inference_pipeline.py --audit_log.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/dependency_error_propagation"),
        help="Directory for audit outputs.",
    )
    args = parser.parse_args()

    raw_rows = read_csv(args.trajectory_csv)
    rows: List[Dict[str, object]] = []
    for row in raw_rows:
        rows.append(
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "gt_joint_type": row["gt_joint_type"],
                "pred_joint_type": row["pred_joint_type"],
                "type_correct": parse_bool(row["joint_type_correct"]),
                "axis_error_deg": parse_float(row["axis_error_deg"]),
                "origin_error_normalized": parse_float(row["origin_error_normalized"]),
                "success_strict": parse_bool(row["success_strict"]),
                "success_relaxed": parse_bool(row["success_relaxed"]),
                "success_axis_only": parse_bool(row["success_axis_only"]),
                "normalized_traj_cd_mean": parse_float(row["normalized_traj_cd_mean"]),
                "normalized_traj_cd_final": parse_float(row["normalized_traj_cd_final"]),
                "endpoint_l2_mean": parse_float(row["endpoint_l2_mean"]),
                "endpoint_l2_median": parse_float(row["endpoint_l2_median"]),
                "failure_mode": "",
            }
        )

    for row in rows:
        modes = []
        if not row["type_correct"]:
            modes.append("joint_type_wrong")
        if (row["axis_error_deg"] or 0.0) >= 45.0:
            modes.append("axis_outlier_ge45")
        elif (row["axis_error_deg"] or 0.0) >= 10.0:
            modes.append("axis_ge10")
        if (row["origin_error_normalized"] or 0.0) >= 0.10:
            modes.append("origin_ge010")
        elif (row["origin_error_normalized"] or 0.0) >= 0.05:
            modes.append("origin_ge005")
        row["failure_mode"] = "+".join(modes) if modes else "pass_or_minor"

    segmentation_rows = load_optional_segmentation(args.segmentation_csv)
    llm_audit_rows = load_optional_jsonl(args.llm_audit_jsonl)
    if segmentation_rows is not None:
        seg_by_case = {row["case_id"]: row for row in parse_segmentation_rows(segmentation_rows)}
        for row in rows:
            seg = seg_by_case.get(row["case_id"])
            row["mask_iou"] = None if seg is None else seg["iou"]
            row["mask_precision"] = None if seg is None else seg["precision"]
            row["mask_recall"] = None if seg is None else seg["recall"]
            row["mask_failure_lt_0p25"] = False if seg is None else ((seg["iou"] or 0.0) < 0.25)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "dependency_error_propagation_cases.csv"
    fieldnames = [
        "case_id",
        "category",
        "gt_joint_type",
        "pred_joint_type",
        "type_correct",
        "axis_error_deg",
        "origin_error_normalized",
        "success_strict",
        "success_relaxed",
        "success_axis_only",
        "normalized_traj_cd_mean",
        "normalized_traj_cd_final",
        "endpoint_l2_mean",
        "endpoint_l2_median",
        "failure_mode",
    ]
    if segmentation_rows is not None:
        fieldnames += ["mask_iou", "mask_precision", "mask_recall", "mask_failure_lt_0p25"]
    write_csv(output_csv, rows, fieldnames)

    report = make_markdown_report(rows, segmentation_rows, llm_audit_rows, output_csv)
    report_path = args.output_dir / "dependency_error_propagation_report.md"
    report_path.write_text(report)

    segmentation_summary = None
    if segmentation_rows is not None:
        parsed_seg = parse_segmentation_rows(segmentation_rows)
        segmentation_summary = {
            "cases": len(parsed_seg),
            "iou": summarize(row["iou"] for row in parsed_seg),
            "precision": summarize(row["precision"] for row in parsed_seg),
            "recall": summarize(row["recall"] for row in parsed_seg),
            "low_iou_lt_0p25": sum((row["iou"] or 0.0) < 0.25 for row in parsed_seg),
            "high_iou_ge_0p50": sum((row["iou"] or 0.0) >= 0.50 for row in parsed_seg),
        }

    summary = {
        "cases": len(rows),
        "type_correct": sum(bool(row["type_correct"]) for row in rows),
        "strict_success": sum(bool(row["success_strict"]) for row in rows),
        "relaxed_success": sum(bool(row["success_relaxed"]) for row in rows),
        "axis_only_success": sum(bool(row["success_axis_only"]) for row in rows),
        "axis_error_deg": summarize(row["axis_error_deg"] for row in rows),
        "origin_error_normalized": summarize(row["origin_error_normalized"] for row in rows),
        "normalized_traj_cd_final": summarize(row["normalized_traj_cd_final"] for row in rows),
        "segmentation_metrics_supplied": segmentation_rows is not None,
        "segmentation": segmentation_summary,
        "llm_audit_rows_supplied": llm_audit_rows is not None,
    }
    (args.output_dir / "dependency_error_propagation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
