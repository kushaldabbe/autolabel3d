"""Bird's Eye View (BEV) visualization for 3D bounding boxes.

BEV projects 3D boxes onto the ground plane (XZ in camera frame), giving a
top-down view of the scene. This is the standard visualization for 3D AV
detection results.

Coordinate mapping (camera frame → BEV pixels):
    u = (X + side_range) × resolution
    v = (range − Z) × resolution    (flip Z so forward = image-up)
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import BBox3D, ObjectClass

BEV_CLASS_COLORS: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.CAR:          (0, 200, 0),
    ObjectClass.PEDESTRIAN:   (0, 0, 200),
    ObjectClass.CYCLIST:      (200, 100, 0),
    ObjectClass.TRAFFIC_CONE: (0, 200, 200),
}

DEFAULT_RANGE_M      = 60.0   # forward range (metres)
DEFAULT_SIDE_RANGE_M = 30.0   # lateral range (metres)
DEFAULT_RESOLUTION   = 10.0   # pixels per metre


def draw_bev(
    boxes: list[BBox3D],
    range_m: float = DEFAULT_RANGE_M,
    side_range_m: float = DEFAULT_SIDE_RANGE_M,
    resolution: float = DEFAULT_RESOLUTION,
    gt_boxes: list[BBox3D] | None = None,
) -> NDArray[np.uint8]:
    """Render a BEV image of 3D bounding boxes.

    Args:
        boxes:       Predicted 3D boxes.
        range_m:     Maximum forward distance to display.
        side_range_m: Maximum lateral distance.
        resolution:  Pixels per metre.
        gt_boxes:    Optional GT boxes (drawn in grey, under predictions).

    Returns:
        (H, W, 3) BGR BEV image.
    """
    W = int(2 * side_range_m * resolution)
    H = int(range_m * resolution)
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    _draw_grid(bev, range_m, side_range_m, resolution)

    ego_u, ego_v = W // 2, H - 1
    cv2.circle(bev, (ego_u, ego_v), 5, (255, 255, 255), -1)
    cv2.putText(bev, "EGO", (ego_u - 15, ego_v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if gt_boxes:
        for box in gt_boxes:
            _draw_bev_box(bev, box, range_m, side_range_m, resolution,
                          color=(100, 100, 100), thickness=1)

    for box in boxes:
        color = BEV_CLASS_COLORS.get(box.class_name, (200, 200, 200))
        _draw_bev_box(bev, box, range_m, side_range_m, resolution,
                      color=color, thickness=2)

    return bev


def _world_to_bev_pixel(
    x: float, z: float,
    range_m: float, side_range_m: float, resolution: float,
    W: int, H: int,
) -> tuple[int, int]:
    u = int((x + side_range_m) * resolution)
    v = int((range_m - z) * resolution)
    return max(0, min(u, W - 1)), max(0, min(v, H - 1))


def _draw_bev_box(
    img: NDArray[np.uint8],
    box: BBox3D,
    range_m: float, side_range_m: float, resolution: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    H, W = img.shape[:2]
    cx, _, cz = box.center
    if cz < 0 or cz > range_m or abs(cx) > side_range_m:
        return

    w, l = box.width / 2, box.length / 2
    cos_y, sin_y = np.cos(box.rotation_y), np.sin(box.rotation_y)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

    corners_local = np.array([[-w, -l], [+w, -l], [+w, +l], [-w, +l]])
    corners_world = (R @ corners_local.T).T + np.array([cx, cz])

    pts = np.array([
        _world_to_bev_pixel(c[0], c[1], range_m, side_range_m, resolution, W, H)
        for c in corners_world
    ], dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    center_px = _world_to_bev_pixel(cx, cz, range_m, side_range_m, resolution, W, H)
    front_px = pts[2:4].mean(axis=0).astype(int)
    cv2.arrowedLine(img, center_px, tuple(front_px), color=color, thickness=1, tipLength=0.3)

    dist = np.hypot(cx, cz)
    cv2.putText(img, f"{box.class_name.value} {dist:.0f}m",
                (center_px[0] + 5, center_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def _draw_grid(
    img: NDArray[np.uint8],
    range_m: float, side_range_m: float, resolution: float,
    grid_spacing: float = 10.0,
) -> None:
    H, W = img.shape[:2]
    grid_color = (30, 30, 30)
    text_color = (80, 80, 80)

    z = grid_spacing
    while z <= range_m:
        v = int((range_m - z) * resolution)
        cv2.line(img, (0, v), (W, v), grid_color, 1)
        cv2.putText(img, f"{z:.0f}m", (5, v - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        z += grid_spacing
