"""Shared normalization utilities for KPP joint-parameter prediction.

KPP predicts a joint origin in the same normalized coordinate frame as the
input point cloud.  The frame must be reproducible at inference time, so it
cannot be centered at the GT joint origin.  Use object bbox center + max side
scale consistently for training and inference.
"""

from __future__ import annotations

import numpy as np


KPP_ORIGIN_NORMALIZATION = "bbox_center_max_side_v1"


def bbox_center_scale(bounds_min, bounds_max):
    """Return bbox center and max-side scale from min/max bounds."""
    bounds_min = np.asarray(bounds_min, dtype=np.float32)
    bounds_max = np.asarray(bounds_max, dtype=np.float32)
    center = (bounds_min + bounds_max) / 2.0
    scale = float(np.max(bounds_max - bounds_min))
    if scale < 1e-6:
        scale = 1.0
    return center.astype(np.float32), scale


def mesh_center_scale(mesh):
    """Return bbox center/scale for a trimesh-like object with `.bounds`."""
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    return bbox_center_scale(bounds[0], bounds[1])


def normalize_points(points, center, scale):
    return ((np.asarray(points, dtype=np.float32) - center) / scale).astype(np.float32)


def normalize_vector(vector, scale):
    return (np.asarray(vector, dtype=np.float32) / scale).astype(np.float32)


def denormalize_points(points, center, scale):
    return (np.asarray(points, dtype=np.float32) * scale + center).astype(np.float32)
