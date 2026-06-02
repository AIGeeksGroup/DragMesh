#!/usr/bin/env python3
"""Sanity checks for KPP origin normalization.

This is intentionally lightweight: it verifies the shared bbox-center/max-side
normalization round trip and catches the old bug where centering at GT origin
would make every origin label exactly zero.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dragmesh.utils.kpp_normalization import (  # noqa: E402
    KPP_ORIGIN_NORMALIZATION,
    bbox_center_scale,
    denormalize_points,
    normalize_points,
)


def main():
    bounds_min = np.array([-1.0, -0.5, -0.25], dtype=np.float32)
    bounds_max = np.array([2.0, 0.75, 1.25], dtype=np.float32)
    gt_origin = np.array([1.4, -0.2, 0.9], dtype=np.float32)

    center, scale = bbox_center_scale(bounds_min, bounds_max)
    origin_norm = normalize_points(gt_origin, center, scale)
    origin_roundtrip = denormalize_points(origin_norm, center, scale)

    if np.allclose(origin_norm, 0.0):
        raise AssertionError("KPP origin label collapsed to zero; check normalization center.")
    if not np.allclose(origin_roundtrip, gt_origin, atol=1e-6):
        raise AssertionError(f"Round trip failed: {origin_roundtrip} != {gt_origin}")

    print(f"OK: {KPP_ORIGIN_NORMALIZATION} round trip passed")
    print(f"center={center.tolist()} scale={scale:.6f} origin_norm={origin_norm.tolist()}")


if __name__ == "__main__":
    main()
