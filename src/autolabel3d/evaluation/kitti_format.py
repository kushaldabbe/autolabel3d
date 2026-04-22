"""KITTI annotation format reader and writer.

KITTI FORMAT (one line per object):
    type truncated occluded alpha x1 y1 x2 y2 height width length x y_bottom z rotation_y [score]

Key conventions:
    - Dimensions order: Height Width Length (HWL, not WHL)
    - Location: bottom centre of the 3D box (y = center_y + height/2)
    - Alpha (observation angle): rotation_y − atan2(z, x)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import BBox3D, ObjectClass
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

OBJECT_CLASS_TO_KITTI: dict[ObjectClass, str] = {
    ObjectClass.CAR:          "Car",
    ObjectClass.PEDESTRIAN:   "Pedestrian",
    ObjectClass.CYCLIST:      "Cyclist",
    ObjectClass.TRAFFIC_CONE: "Misc",
}

KITTI_TO_OBJECT_CLASS: dict[str, ObjectClass] = {
    "Car":            ObjectClass.CAR,
    "Van":            ObjectClass.CAR,
    "Truck":          ObjectClass.CAR,
    "Pedestrian":     ObjectClass.PEDESTRIAN,
    "Person_sitting": ObjectClass.PEDESTRIAN,
    "Cyclist":        ObjectClass.CYCLIST,
    "Tram":           ObjectClass.CAR,
    "Misc":           ObjectClass.TRAFFIC_CONE,
}


def write_kitti_annotations(
    boxes: list[BBox3D],
    output_path: Path,
    bbox_2d: NDArray[np.float32] | None = None,
) -> None:
    """Write 3D boxes to a KITTI-format .txt file.

    Args:
        boxes: 3D boxes to write.
        output_path: Destination .txt file (parent dirs created automatically).
        bbox_2d: Optional (N, 4) 2D boxes [x1,y1,x2,y2]. Defaults to zeros.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, box in enumerate(boxes):
        kitti_type = OBJECT_CLASS_TO_KITTI.get(box.class_name, "Misc")

        # Alpha: observation angle accounts for camera-to-object ray
        alpha = box.rotation_y - np.arctan2(box.center[2], box.center[0])
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # clamp to [-π, π]

        x1, y1, x2, y2 = (
            tuple(bbox_2d[i]) if (bbox_2d is not None and i < len(bbox_2d))
            else (0.0, 0.0, 0.0, 0.0)
        )

        h, w, l = box.height, box.width, box.length
        x, y, z = box.center
        y_bottom = y + h / 2  # geometric centre → bottom centre

        lines.append(
            f"{kitti_type} 0.00 0 {alpha:.2f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{x:.2f} {y_bottom:.2f} {z:.2f} {box.rotation_y:.2f} {box.confidence:.4f}"
        )

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    logger.debug("Wrote %d annotations to %s", len(lines), output_path)


def read_kitti_annotations(
    filepath: Path,
    classes: set[ObjectClass] | None = None,
) -> list[BBox3D]:
    """Read 3D boxes from a KITTI-format .txt file.

    Args:
        filepath: Source .txt file.
        classes: Filter to these classes. None = all.

    Returns:
        List of BBox3D objects.
    """
    if not filepath.exists():
        logger.warning("Annotation file not found: %s", filepath)
        return []

    text = filepath.read_text().strip()
    if not text:
        return []

    boxes: list[BBox3D] = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 15:
            logger.warning("Malformed line %d in %s (got %d fields)", line_num, filepath, len(parts))
            continue

        obj_class = KITTI_TO_OBJECT_CLASS.get(parts[0])
        if obj_class is None:
            continue
        if classes is not None and obj_class not in classes:
            continue

        h, w, l  = float(parts[8]), float(parts[9]), float(parts[10])
        x        = float(parts[11])
        y_bottom = float(parts[12])
        z        = float(parts[13])
        ry       = float(parts[14])
        score    = float(parts[15]) if len(parts) > 15 else 1.0

        boxes.append(BBox3D(
            center=np.array([x, y_bottom - h / 2, z], dtype=np.float64),
            dimensions=np.array([w, h, l], dtype=np.float64),
            rotation_y=ry,
            class_name=obj_class,
            confidence=score,
        ))

    logger.debug("Read %d annotations from %s", len(boxes), filepath)
    return boxes
