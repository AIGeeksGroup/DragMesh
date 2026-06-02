#!/usr/bin/env python3
"""Export GAP-25 meshes and GT movable masks for P3-SAM evaluation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dragmesh.data.data_loader_v2 import GAPartNetLoaderV2  # noqa: E402


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def object_id_from_case(case_id: str) -> str:
    return case_id.rsplit("_", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases_csv",
        type=Path,
        default=Path("/home/data2/zhangaho/1/zhangh/results/dependency_error_propagation/dependency_error_propagation_cases.csv"),
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=ROOT / "data/gapartnet_full/partnet_mobility_part",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/data2/zhangaho/1/zhangh/results/p3sam_gap25_assets"),
    )
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for row in read_csv(args.cases_csv):
        case_id = row["case_id"]
        object_id = object_id_from_case(case_id)
        obj_dir = args.dataset_root / object_id
        if not obj_dir.exists():
            raise FileNotFoundError(f"Missing GAPartNet object dir for {case_id}: {obj_dir}")

        loader = GAPartNetLoaderV2(str(obj_dir))
        sample = loader.generate_training_sample(0, num_frames=args.num_frames, return_mesh=True)
        mesh = sample["initial_mesh"]
        vertex_mask = np.asarray(sample["part_mask"]).astype(bool)
        face_mask = vertex_mask[mesh.faces].mean(axis=1) > 0.5

        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = case_dir / f"{case_id}.glb"
        gt_mask_path = case_dir / f"{case_id}_gt_movable_face_mask.npy"
        mesh.export(mesh_path)
        np.save(gt_mask_path, face_mask.astype(np.uint8))

        manifest_rows.append(
            {
                "case_id": case_id,
                "category": row["category"],
                "object_id": object_id,
                "mesh_path": mesh_path,
                "gt_mask_path": gt_mask_path,
                "gt_mask_format": "binary_face",
                "gt_joint_type": row["gt_joint_type"],
            }
        )

    manifest_path = args.output_dir / "gap25_p3sam_assets_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "case_id",
            "category",
            "object_id",
            "mesh_path",
            "gt_mask_path",
            "gt_mask_format",
            "gt_joint_type",
        ],
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
