"""Comparison visualization — GT vs. predicted 3D boxes.

Produces BEV overlay and side-by-side panels for qualitative evaluation.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import BBox3D, ObjectClass
from autolabel3d.visualization.bev import draw_bev

GT_COLOR: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.CAR:          (200, 100, 0),
    ObjectClass.PEDESTRIAN:   (200, 0, 100),
    ObjectClass.CYCLIST:      (200, 150, 0),
    ObjectClass.TRAFFIC_CONE: (200, 200, 0),
}

PRED_COLOR: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.CAR:          (0, 200, 0),
    ObjectClass.PEDESTRIAN:   (0, 0, 200),
    ObjectClass.CYCLIST:      (0, 200, 200),
    ObjectClass.TRAFFIC_CONE: (0, 255, 255),
}


def draw_comparison_bev(
    pred_boxes: list[BBox3D],
    gt_boxes: list[BBox3D],
    range_m: float = 60.0,
    side_range_m: float = 30.0,
    resolution: float = 10.0,
) -> NDArray[np.uint8]:
    """BEV overlay with GT (grey) and predicted (color) boxes."""
    return draw_bev(
        pred_boxes,
        range_m=range_m,
        side_range_m=side_range_m,
        resolution=resolution,
        gt_boxes=gt_boxes,
    )


def draw_comparison_image(
    image: NDArray[np.uint8],
    pred_boxes: list[BBox3D],
    gt_boxes: list[BBox3D],
    intrinsics_matrix: NDArray[np.float64] | None = None,
) -> NDArray[np.uint8]:
    """Project 3D box wireframes onto the image (GT = blue, Pred = green).

    Args:
        image: (H, W, 3) BGR image.
        pred_boxes: Predicted boxes drawn in green.
        gt_boxes: GT boxes drawn in blue.
        intrinsics_matrix: (3, 3) camera matrix K. If None, only shows counts.

    Returns:
        (H, W, 3) annotated BGR image.
    """
    vis = image.copy()

    if intrinsics_matrix is None:
        cv2.putText(
            vis, f"Pred: {len(pred_boxes)} | GT: {len(gt_boxes)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        return vis

    for box in gt_boxes:
        _draw_projected_box(vis, box, intrinsics_matrix, GT_COLOR, thickness=2)
    for box in pred_boxes:
        _draw_projected_box(vis, box, intrinsics_matrix, PRED_COLOR, thickness=2)

    cv2.putText(vis, "GT",   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)
    cv2.putText(vis, "Pred", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0),   2)
    return vis


def draw_side_by_side(
    image: NDArray[np.uint8],
    pred_boxes: list[BBox3D],
    gt_boxes: list[BBox3D],
    range_m: float = 60.0,
) -> NDArray[np.uint8]:
    """Horizontal composite: camera image (left) + BEV (right).

    The BEV is auto-scaled to match the camera image height.

    Returns:
        (H, W_cam + W_bev, 3) combined BGR image.
    """
    H, W = image.shape[:2]
    resolution = H / range_m
    side_range = range_m / 2

    bev = draw_bev(pred_boxes, range_m=range_m,
                   side_range_m=side_range, resolution=resolution,
                   gt_boxes=gt_boxes)
    bev_resized = cv2.resize(bev, (bev.shape[1], H))

    combined = np.concatenate([image, bev_resized], axis=1)
    cv2.line(combined, (W, 0), (W, H - 1), (255, 255, 255), 2)
    return combined


def _draw_projected_box(
    image: NDArray[np.uint8],
    box: BBox3D,
    K: NDArray[np.float64],
    color_map: dict[ObjectClass, tuple[int, int, int]],
    thickness: int = 2,
) -> None:
    corners_3d = box.corners()  # (8, 3)
    if np.any(corners_3d[:, 2] <= 0):
        return

    projected = (K @ corners_3d.T).T  # (8, 3)
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    pts = projected[:, :2].astype(int)

    H, W = image.shape[:2]
    color = color_map.get(box.class_name, (200, 200, 200))
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
    ]
    for i, j in edges:
        p1, p2 = pts[i], pts[j]
        if (
            -W <= p1[0] <= 2 * W and -H <= p1[1] <= 2 * H
            and -W <= p2[0] <= 2 * W and -H <= p2[1] <= 2 * H
        ):
            cv2.line(image, tuple(p1), tuple(p2), color, thickness)
