#!/usr/bin/env python3
"""Evaluate P3-SAM or other part masks against binary movable-part labels.

The script is intentionally format-lightweight: provide a manifest CSV with one
row per case and paths to predicted and GT masks. Multi-part predicted labels
are matched to the GT movable mask by best IoU unless `pred_part_id` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh


LABEL_FORMATS = {"face_labels", "vertex_labels"}
BINARY_FORMATS = {"binary_face", "binary_vertex"}
ALL_FORMATS = LABEL_FORMATS | BINARY_FORMATS


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "mask" in data:
            return data["mask"]
        if "face_ids" in data:
            return data["face_ids"]
        first_key = list(data.keys())[0]
        return data[first_key]
    raise ValueError(f"Unsupported mask file extension: {path}")


def face_labels_to_vertex_labels(faces: np.ndarray, num_vertices: int, face_labels: np.ndarray) -> np.ndarray:
    vertex_votes: List[List[int]] = [[] for _ in range(num_vertices)]
    for fid, face in enumerate(faces):
        label = int(face_labels[fid])
        if label < 0:
            continue
        for vid in face:
            vertex_votes[int(vid)].append(label)

    vertex_labels = -np.ones(num_vertices, dtype=np.int64)
    for vid, labels in enumerate(vertex_votes):
        if labels:
            counts = np.bincount(np.asarray(labels, dtype=np.int64))
            vertex_labels[vid] = int(np.argmax(counts))
    return vertex_labels


def vertex_binary_to_face_binary(faces: np.ndarray, vertex_mask: np.ndarray) -> np.ndarray:
    return vertex_mask[faces].mean(axis=1) > 0.5


def vertex_labels_to_face_labels(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    face_labels = -np.ones(len(faces), dtype=np.int64)
    for fid, face in enumerate(faces):
        labels = vertex_labels[face]
        labels = labels[labels >= 0]
        if labels.size:
            counts = np.bincount(labels.astype(np.int64))
            face_labels[fid] = int(np.argmax(counts))
    return face_labels


def as_face_array(mask: np.ndarray, mask_format: str, mesh: trimesh.Trimesh) -> np.ndarray:
    if mask_format not in ALL_FORMATS:
        raise ValueError(f"Unknown mask format {mask_format!r}; expected one of {sorted(ALL_FORMATS)}")
    values = np.asarray(mask).reshape(-1)
    if mask_format in {"face_labels", "binary_face"}:
        if values.shape[0] != len(mesh.faces):
            raise ValueError(f"Face mask length {values.shape[0]} != mesh faces {len(mesh.faces)}")
        return values
    if values.shape[0] != len(mesh.vertices):
        raise ValueError(f"Vertex mask length {values.shape[0]} != mesh vertices {len(mesh.vertices)}")
    if mask_format == "binary_vertex":
        return vertex_binary_to_face_binary(mesh.faces, values.astype(bool))
    return vertex_labels_to_face_labels(mesh.faces, values.astype(np.int64))


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)
    tp = float(np.logical_and(pred_bool, gt_bool).sum())
    fp = float(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = float(np.logical_and(~pred_bool, gt_bool).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return precision, recall, iou


def best_pred_mask(pred_face: np.ndarray, pred_format: str, gt_face_binary: np.ndarray,
                   pred_part_id: Optional[int]) -> Tuple[np.ndarray, Optional[int], float]:
    if pred_format in BINARY_FORMATS:
        pred_binary = pred_face.astype(bool)
        _, _, iou = binary_metrics(pred_binary, gt_face_binary)
        return pred_binary, pred_part_id, iou

    labels = [int(v) for v in np.unique(pred_face) if int(v) >= 0]
    if pred_part_id is not None:
        pred_binary = pred_face.astype(np.int64) == int(pred_part_id)
        _, _, iou = binary_metrics(pred_binary, gt_face_binary)
        return pred_binary, int(pred_part_id), iou

    best_label = None
    best_iou = -1.0
    best_binary = np.zeros_like(gt_face_binary, dtype=bool)
    for label in labels:
        candidate = pred_face.astype(np.int64) == label
        _, _, iou = binary_metrics(candidate, gt_face_binary)
        if iou > best_iou:
            best_label = label
            best_iou = iou
            best_binary = candidate
    return best_binary, best_label, max(best_iou, 0.0)


def oracle_union_pred_mask(pred_face: np.ndarray, pred_format: str,
                           gt_face_binary: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """Greedily groups predicted P3-SAM parts into a movable/rest binary mask."""
    if pred_format in BINARY_FORMATS:
        pred_binary = pred_face.astype(bool)
        _, _, iou = binary_metrics(pred_binary, gt_face_binary)
        return pred_binary, "binary", iou

    labels = [int(v) for v in np.unique(pred_face) if int(v) >= 0]
    selected: List[int] = []
    current = np.zeros_like(gt_face_binary, dtype=bool)
    _, _, current_iou = binary_metrics(current, gt_face_binary)

    while True:
        best_label = None
        best_mask = current
        best_iou = current_iou
        for label in labels:
            if label in selected:
                continue
            candidate = np.logical_or(current, pred_face.astype(np.int64) == label)
            _, _, iou = binary_metrics(candidate, gt_face_binary)
            if iou > best_iou + 1e-12:
                best_label = label
                best_mask = candidate
                best_iou = iou
        if best_label is None:
            break
        selected.append(best_label)
        current = best_mask
        current_iou = best_iou

    return current, ";".join(str(v) for v in selected), current_iou


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV with case_id, mesh_path, pred_mask_path, gt_mask_path, pred_mask_format, gt_mask_format.")
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument(
        "--selection_mode",
        choices=["best_single", "oracle_union"],
        default="best_single",
        help="best_single evaluates one predicted part; oracle_union groups predicted parts into movable/rest.",
    )
    args = parser.parse_args()

    rows = []
    for src in read_csv(args.manifest):
        mesh = trimesh.load(src["mesh_path"], process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        pred_format = src.get("pred_mask_format", "face_labels")
        gt_format = src.get("gt_mask_format", "binary_face")
        pred_face = as_face_array(load_mask(Path(src["pred_mask_path"])), pred_format, mesh)
        gt_face = as_face_array(load_mask(Path(src["gt_mask_path"])), gt_format, mesh).astype(bool)
        pred_part_id = src.get("pred_part_id", "").strip()
        pred_part = int(pred_part_id) if pred_part_id else None

        if args.selection_mode == "oracle_union" and pred_part is None:
            pred_binary, matched_part, _ = oracle_union_pred_mask(pred_face, pred_format, gt_face)
        else:
            pred_binary, matched_part, _ = best_pred_mask(pred_face, pred_format, gt_face, pred_part)
        precision, recall, iou = binary_metrics(pred_binary, gt_face)
        rows.append({
            "case_id": src.get("case_id", Path(src["mesh_path"]).stem),
            "matched_pred_part_id": "" if matched_part is None else matched_part,
            "selection_mode": args.selection_mode,
            "oracle_selection": args.selection_mode in {"best_single", "oracle_union"} and pred_part is None,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "pred_positive_faces": int(pred_binary.sum()),
            "gt_positive_faces": int(gt_face.sum()),
        })

    fieldnames = [
        "case_id",
        "matched_pred_part_id",
        "selection_mode",
        "oracle_selection",
        "precision",
        "recall",
        "iou",
        "pred_positive_faces",
        "gt_positive_faces",
    ]
    write_csv(args.output_csv, rows, fieldnames)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "cases": len(rows),
            "selection_mode": args.selection_mode,
            "oracle_selection": args.selection_mode in {"best_single", "oracle_union"},
            "precision": summarize(float(row["precision"]) for row in rows),
            "recall": summarize(float(row["recall"]) for row in rows),
            "iou": summarize(float(row["iou"]) for row in rows),
        }
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {args.output_csv}")
    if args.summary_json:
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
