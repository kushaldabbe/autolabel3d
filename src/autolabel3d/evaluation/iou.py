"""3D Intersection over Union for oriented bounding boxes.

IoU_3D = V_intersection / V_union = V_intersection / (V_A + V_B - V_intersection)

For ORIENTED boxes (rotated around Y-axis), we decompose into:
    1. BEV IoU — 2D polygon intersection on the XZ ground plane
       (Sutherland-Hodgman clipping algorithm + shoelace area formula)
    2. Height overlap — 1D overlap along Y-axis

    V_intersection = Area_BEV × height_overlap
    V_union = V_A + V_B - V_intersection
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import BBox3D


def compute_iou_3d(box_a: BBox3D, box_b: BBox3D) -> float:
    """3D IoU between two oriented bounding boxes (decomposed into BEV + height)."""
    corners_a = _get_bev_corners(box_a)
    corners_b = _get_bev_corners(box_b)
    bev_area = _polygon_intersection_area(corners_a, corners_b)

    if bev_area <= 0:
        return 0.0

    a_y_min = box_a.center[1] - box_a.height / 2
    a_y_max = box_a.center[1] + box_a.height / 2
    b_y_min = box_b.center[1] - box_b.height / 2
    b_y_max = box_b.center[1] + box_b.height / 2
    y_overlap = max(0.0, min(a_y_max, b_y_max) - max(a_y_min, b_y_min))

    if y_overlap <= 0:
        return 0.0

    inter_vol = bev_area * y_overlap
    union_vol = box_a.volume + box_b.volume - inter_vol
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def compute_iou_3d_batch(
    boxes_pred: list[BBox3D], boxes_gt: list[BBox3D]
) -> NDArray[np.float64]:
    """Compute (M, N) IoU matrix between M predicted and N GT boxes."""
    M, N = len(boxes_pred), len(boxes_gt)
    iou = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            iou[i, j] = compute_iou_3d(boxes_pred[i], boxes_gt[j])
    return iou


def compute_iou_bev(box_a: BBox3D, box_b: BBox3D) -> float:
    """BEV (Bird's Eye View) 2D IoU — ignores height, only XZ footprint."""
    corners_a = _get_bev_corners(box_a)
    corners_b = _get_bev_corners(box_b)
    inter = _polygon_intersection_area(corners_a, corners_b)
    union = box_a.width * box_a.length + box_b.width * box_b.length - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_bev_corners(box: BBox3D) -> NDArray[np.float64]:
    """Return 4 (x, z) ground-plane corners of a 3D box (CCW order)."""
    cx, _, cz = box.center
    w, l = box.width / 2, box.length / 2
    cos, sin = np.cos(box.rotation_y), np.sin(box.rotation_y)

    # Local corners (counter-clockwise)
    corners_local = np.array([[-w, -l], [+w, -l], [+w, +l], [-w, +l]], dtype=np.float64)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    return (R @ corners_local.T).T + np.array([cx, cz])


def _polygon_intersection_area(
    poly_a: NDArray[np.float64],
    poly_b: NDArray[np.float64],
) -> float:
    """Area of intersection of two convex polygons via Sutherland-Hodgman clipping.

    For each edge of poly_b, clip poly_a against the half-plane.
    Result is the intersection polygon; area computed by the shoelace formula.
    """
    output = poly_a.tolist()
    if not output:
        return 0.0

    n = len(poly_b)
    for i in range(n):
        if not output:
            return 0.0
        input_list = output
        output = []
        edge_s = poly_b[i]
        edge_e = poly_b[(i + 1) % n]

        for j in range(len(input_list)):
            cur = np.array(input_list[j])
            prev = np.array(input_list[j - 1])
            cur_in = _cross_2d(edge_s, edge_e, cur) >= 0
            prev_in = _cross_2d(edge_s, edge_e, prev) >= 0

            if cur_in:
                if not prev_in:
                    pt = _line_intersection(edge_s, edge_e, prev, cur)
                    if pt is not None:
                        output.append(pt.tolist())
                output.append(cur.tolist())
            elif prev_in:
                pt = _line_intersection(edge_s, edge_e, prev, cur)
                if pt is not None:
                    output.append(pt.tolist())

    return _shoelace_area(np.array(output)) if len(output) >= 3 else 0.0


def _cross_2d(o: NDArray, a: NDArray, b: NDArray) -> float:
    """Signed 2D cross product (a-o) × (b-o). Positive = b is left of o→a."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _line_intersection(
    p1: NDArray, p2: NDArray, p3: NDArray, p4: NDArray
) -> NDArray | None:
    """Intersection of line p1-p2 with line p3-p4 (parametric form)."""
    d1, d2 = p2 - p1, p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    d3 = p3 - p1
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / denom
    return p1 + t * d1


def _shoelace_area(polygon: NDArray[np.float64]) -> float:
    """Area of a polygon using the shoelace formula: A = ½|Σ(xᵢyᵢ₊₁ − xᵢ₊₁yᵢ)|."""
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
