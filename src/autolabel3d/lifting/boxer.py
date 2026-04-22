"""Boxer: geometry-only 2D → 3D lifting using camera model + size priors.

NO DEPTH NETWORK REQUIRED.

THE CORE GEOMETRIC INSIGHT — triangle similarity:

    y_pixels = fy × H_real / Z
    ⟹ Z = fy × H_real / y_pixels

where:
    y_pixels  = 2D box height in pixels
    fy        = focal length (y-direction, pixels)
    H_real    = real-world object height (metres) — the class prior
    Z         = depth from camera (metres) — what we solve for

GROUND PLANE CONSTRAINT:
The bottom of the 2D box is the object's ground contact. With camera height
above ground = h_cam:
    Y_cam = h_cam − H_real/2   (box centre, camera Y is down)

Combined with Z from the formula above, X from horizontal pixel offset, and
a ray-angle heading estimate, we get a full 3D box with no neural network.

LIMITATIONS:
    - Accuracy depends on size prior quality
    - Ground plane assumption fails on slopes
    - Heading is a rough approximation (no appearance cue)
    - Useful as a fast baseline or lightweight fallback
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import (
    BBox3D,
    CameraCalibration,
    ObjectClass,
    SegmentationMask,
)
from autolabel3d.lifting.base import BaseLifter
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SIZE_PRIORS: dict[ObjectClass, tuple[float, float, float]] = {
    ObjectClass.CAR:          (1.8, 1.5, 4.5),
    ObjectClass.PEDESTRIAN:   (0.6, 1.7, 0.6),
    ObjectClass.CYCLIST:      (0.6, 1.7, 1.8),
    ObjectClass.TRAFFIC_CONE: (0.3, 0.8, 0.3),
}


class BoxerLifter(BaseLifter):
    """Lightweight geometric 2D → 3D lifter. No GPU, no model weights.

    Uses triangle similarity distance + optional ground-plane constraint.
    Runs in microseconds per object; strong baseline for ablation studies.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

        fitting_cfg = getattr(cfg, "fitting", None)
        self.camera_height: float = (
            getattr(fitting_cfg, "camera_height", 1.65) if fitting_cfg else 1.65
        )
        self.use_ground_plane: bool = (
            getattr(fitting_cfg, "use_ground_plane", True) if fitting_cfg else True
        )

        self.size_priors: dict[ObjectClass, tuple[float, float, float]] = dict(
            DEFAULT_SIZE_PRIORS
        )
        if fitting_cfg:
            for class_name in ("car", "pedestrian", "cyclist", "traffic_cone"):
                try:
                    cls = ObjectClass(class_name)
                    w, h, l = self.size_priors[cls]
                    hp = getattr(fitting_cfg, "height_prior", None)
                    wp = getattr(fitting_cfg, "width_prior", None)
                    lp = getattr(fitting_cfg, "length_prior", None)
                    if hp and hasattr(hp, class_name):
                        h = float(getattr(hp, class_name))
                    if wp and hasattr(wp, class_name):
                        w = float(getattr(wp, class_name))
                    if lp and hasattr(lp, class_name):
                        l = float(getattr(lp, class_name))
                    self.size_priors[cls] = (w, h, l)
                except ValueError:
                    pass

        logger.info(
            "BoxerLifter: camera_height=%.2f, ground_plane=%s",
            self.camera_height, self.use_ground_plane,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _estimate_distance(
        self, bbox: NDArray[np.float32], fy: float, h_prior: float
    ) -> float:
        """Triangle similarity: Z = fy × H_real / h_pixels."""
        h_pixels = max(float(bbox[3] - bbox[1]), 1.0)
        return fy * h_prior / h_pixels

    def _estimate_heading(self, bbox: NDArray[np.float32], cx: float) -> float:
        """Approximate yaw from horizontal image position (ray angle heuristic)."""
        u_center = (bbox[0] + bbox[2]) / 2.0
        return float(np.arctan2(u_center - cx, 1.0))

    # ------------------------------------------------------------------
    # Lifting
    # ------------------------------------------------------------------

    def lift(
        self,
        mask: SegmentationMask,
        calibration: CameraCalibration,
        image: NDArray[np.uint8] | None = None,
    ) -> BBox3D | None:
        """Lift one mask to a 3D box using geometry only.

        Steps:
            1. Distance Z via triangle similarity.
            2. X from horizontal pixel offset.
            3. Y via ground-plane constraint (or 2D box centre).
            4. Heading from box position.
            5. Camera → world transform.
        """
        intrinsics = calibration.intrinsics
        extrinsics = calibration.extrinsics

        w_prior, h_prior, l_prior = self.size_priors.get(
            mask.class_name, (1.0, 1.0, 1.0)
        )
        bbox = mask.bbox

        Z = float(np.clip(self._estimate_distance(bbox, intrinsics.fy, h_prior), 1.0, 200.0))
        u_center = (bbox[0] + bbox[2]) / 2.0
        X = float((u_center - intrinsics.cx) * Z / intrinsics.fx)

        if self.use_ground_plane:
            v_bottom = float(bbox[3])
            Y_bottom = (v_bottom - intrinsics.cy) * Z / intrinsics.fy
            Y = Y_bottom - h_prior / 2.0
        else:
            v_center = (bbox[1] + bbox[3]) / 2.0
            Y = float((v_center - intrinsics.cy) * Z / intrinsics.fy)

        rotation_y = self._estimate_heading(bbox, intrinsics.cx)

        center_cam = np.array([X, Y, Z], dtype=np.float64)
        center_world = (extrinsics.rotation @ center_cam) + extrinsics.translation

        return BBox3D(
            center=center_world,
            dimensions=np.array([w_prior, h_prior, l_prior], dtype=np.float64),
            rotation_y=rotation_y,
            class_name=mask.class_name,
            confidence=mask.confidence,
            track_id=mask.track_id,
        )

    def lift_batch(
        self,
        masks: list[SegmentationMask],
        calibration: CameraCalibration,
        image: NDArray[np.uint8] | None = None,
    ) -> list[BBox3D | None]:
        results = [self.lift(m, calibration, image) for m in masks]
        logger.info("Boxer lifted %d/%d masks", sum(1 for r in results if r), len(masks))
        return results
