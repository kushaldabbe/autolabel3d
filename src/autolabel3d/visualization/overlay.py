"""2D overlay visualization — draw detections and masks on frames.

All functions take and return BGR images (OpenCV convention) so they compose
naturally with the rest of the pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import FrameDetections, FrameMasks, ObjectClass

CLASS_COLORS: dict[ObjectClass, tuple[int, int, int]] = {
    ObjectClass.CAR:          (0, 255, 0),    # green
    ObjectClass.PEDESTRIAN:   (0, 0, 255),    # red
    ObjectClass.CYCLIST:      (255, 165, 0),  # blue-orange
    ObjectClass.TRAFFIC_CONE: (0, 255, 255),  # yellow
}
DEFAULT_COLOR = (255, 255, 255)


def draw_detections(
    image: NDArray[np.uint8],
    detections: FrameDetections,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> NDArray[np.uint8]:
    """Draw 2D bounding boxes and confidence labels on a copy of the image."""
    vis = image.copy()

    for det in detections.detections:
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = det.bbox.astype(int)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)

        label = f"{det.class_name.value} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return vis


def draw_masks(
    image: NDArray[np.uint8],
    masks: FrameMasks,
    alpha: float = 0.4,
) -> NDArray[np.uint8]:
    """Overlay segmentation masks with alpha blending and contour outlines.

    Alpha blending: output = α × mask_color + (1 − α) × original
    """
    vis = image.copy()

    for seg_mask in masks.masks:
        color = CLASS_COLORS.get(seg_mask.class_name, DEFAULT_COLOR)
        binary = seg_mask.mask  # (H, W) bool

        overlay = np.zeros_like(vis)
        overlay[binary] = color
        vis[binary] = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)[binary]

        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

    return vis
