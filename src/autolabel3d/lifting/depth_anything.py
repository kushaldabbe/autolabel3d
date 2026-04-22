"""Depth Anything V2 lifter: monocular depth → 3D bounding boxes.

MONOCULAR DEPTH — KEY INSIGHTS:

Depth Anything V2 uses a DINOv2 encoder + DPT decoder to predict RELATIVE
(scale-ambiguous) depth from a single RGB image:
    true_depth ≈ s · predicted_depth + b   (unknown s and b)

SCALE RECOVERY VIA SIZE PRIORS:
Since metric scale is unknown, we recover it using known object dimensions.
Example: prior car length = 4.5m; PCA-fitted length = 9m → scale = 0.5.
    metric_depth ≈ scale × relative_depth

PIPELINE PER OBJECT:
    1. Predict dense depth map for the full frame (cached per frame)
    2. Back-project mask region to 3D point cloud via camera intrinsics
    3. Filter outlier depths (|d - median| < 2σ)
    4. Fit oriented 3D bounding box (PCA or ground-plane-aligned)
    5. Recover metric scale using class size priors
    6. Transform box centre camera → world frame
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from autolabel3d.data.schemas import (
    BBox3D,
    CameraCalibration,
    ObjectClass,
    SegmentationMask,
)
from autolabel3d.lifting.base import BaseLifter
from autolabel3d.utils.device import get_device
from autolabel3d.utils.geometry import (
    depth_map_to_pointcloud,
    fit_3d_bbox_min_enclosing,
    fit_3d_bbox_pca,
    transform_points,
)
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HEIGHT_PRIORS: dict[ObjectClass, float] = {
    ObjectClass.CAR:          1.5,
    ObjectClass.PEDESTRIAN:   1.7,
    ObjectClass.CYCLIST:      1.7,
    ObjectClass.TRAFFIC_CONE: 0.8,
}

DEFAULT_SIZE_PRIORS: dict[ObjectClass, tuple[float, float, float]] = {
    ObjectClass.CAR:          (1.8, 1.5, 4.5),
    ObjectClass.PEDESTRIAN:   (0.6, 1.7, 0.6),
    ObjectClass.CYCLIST:      (0.6, 1.7, 1.8),
    ObjectClass.TRAFFIC_CONE: (0.3, 0.8, 0.3),
}


class DepthAnythingLifter(BaseLifter):
    """Lifts 2D masks to 3D boxes using Depth Anything V2 + size priors."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.model_size: str = getattr(cfg, "model_size", "base")
        self.model_id = (
            f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}-hf"
        )

        fitting_cfg = getattr(cfg, "fitting", None)
        self.fitting_method: str = getattr(fitting_cfg, "method", "pca") if fitting_cfg else "pca"
        self.min_points: int = getattr(fitting_cfg, "min_points", 50) if fitting_cfg else 50

        self.height_priors: dict[ObjectClass, float] = dict(DEFAULT_HEIGHT_PRIORS)
        if fitting_cfg and hasattr(fitting_cfg, "height_prior") and fitting_cfg.height_prior:
            for class_name, val in fitting_cfg.height_prior.items():
                try:
                    self.height_priors[ObjectClass(class_name)] = float(val)
                except ValueError:
                    logger.warning("Unknown class in height_prior config: %s", class_name)

        self._model: torch.nn.Module | None = None
        self._processor: Any = None
        self._cached_depth: NDArray[np.float64] | None = None
        self._cached_image_id: int | None = None

        logger.info("DepthAnythingLifter: %s, fitting=%s", self.model_id, self.fitting_method)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        logger.info("Loading Depth Anything V2: %s", self.model_id)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("Depth Anything V2 ready on %s", self.device)

    # ------------------------------------------------------------------
    # Depth prediction
    # ------------------------------------------------------------------

    def predict_depth(self, image: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Predict relative depth map for a BGR image (H, W, 3).

        The model outputs inverse depth (disparity-like: higher = closer).
        We apply reciprocal inversion: depth = 1 / (inv_depth + ε) to
        preserve depth ratios correctly for back-projection.

        Output: (H, W) float64, higher = farther (standard depth convention).
        """
        image_id = id(image)
        if self._cached_image_id == image_id and self._cached_depth is not None:
            return self._cached_depth

        self._load_model()

        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        H, W = image.shape[:2]
        depth_tensor = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth_np = depth_tensor.cpu().numpy().astype(np.float64)
        # Reciprocal inversion: preserves depth ratios (linear flip does not)
        depth_np = 1.0 / (depth_np + 1e-6)

        self._cached_depth = depth_np
        self._cached_image_id = image_id
        return depth_np

    # ------------------------------------------------------------------
    # Lifting
    # ------------------------------------------------------------------

    def _lift_single(
        self,
        mask: SegmentationMask,
        depth_map: NDArray[np.float64],
        calibration: CameraCalibration,
    ) -> BBox3D | None:
        """Lift one segmentation mask to a 3D bounding box."""
        points_cam = depth_map_to_pointcloud(
            depth_map=depth_map,
            intrinsics=calibration.intrinsics,
            mask=mask.mask,
        )

        if len(points_cam) < self.min_points:
            logger.debug("Track %d: %d points < min %d, skipping", mask.track_id, len(points_cam), self.min_points)
            return None

        # Outlier filter: keep depths within 2σ of median
        depths = points_cam[:, 2]
        median_d = np.median(depths)
        std_d = np.std(depths)
        inlier = np.abs(depths - median_d) < 2.0 * max(std_d, 1e-6)
        points_cam = points_cam[inlier]

        if len(points_cam) < self.min_points:
            logger.debug("Track %d: %d inlier points < min %d, skipping", mask.track_id, len(points_cam), self.min_points)
            return None

        height_prior = self.height_priors.get(mask.class_name)

        if self.fitting_method == "min_enclosing":
            center, dimensions, rotation_y = fit_3d_bbox_min_enclosing(
                points_cam, height_prior=height_prior,
            )
        else:
            center, dimensions, rotation_y = fit_3d_bbox_pca(points_cam)

        # Scale recovery using largest-to-largest dimension matching
        size_prior = DEFAULT_SIZE_PRIORS.get(mask.class_name)
        if size_prior is not None:
            prior_sorted = sorted(size_prior, reverse=True)
            fit_sorted = sorted(dimensions, reverse=True)
            scale = float(prior_sorted[0] / max(fit_sorted[0], 1e-6))

            dimensions = dimensions * scale
            center = center * scale

            # Clamp each sorted dimension to [0.3x, 2.5x] of the prior
            clamped = np.array(sorted(dimensions, reverse=True))
            for i, pv in enumerate(prior_sorted):
                clamped[i] = float(np.clip(clamped[i], pv * 0.3, pv * 2.5))

            # Reassign: largest→length, middle→width, smallest→height
            s = sorted(clamped, reverse=True)
            dimensions = np.array([s[1], s[2], s[0]], dtype=np.float64)

        center_world = transform_points(
            center.reshape(1, 3),
            calibration.extrinsics.rotation,
            calibration.extrinsics.translation,
        ).squeeze()

        return BBox3D(
            center=center_world,
            dimensions=dimensions,
            rotation_y=rotation_y,
            class_name=mask.class_name,
            confidence=mask.confidence,
            track_id=mask.track_id,
        )

    def lift(
        self,
        mask: SegmentationMask,
        calibration: CameraCalibration,
        image: NDArray[np.uint8] | None = None,
    ) -> BBox3D | None:
        if image is None:
            raise ValueError("DepthAnythingLifter requires the image.")
        return self._lift_single(mask, self.predict_depth(image), calibration)

    def lift_batch(
        self,
        masks: list[SegmentationMask],
        calibration: CameraCalibration,
        image: NDArray[np.uint8] | None = None,
    ) -> list[BBox3D | None]:
        """Lift all masks in one frame, sharing one depth inference."""
        if image is None:
            raise ValueError("DepthAnythingLifter requires the image.")
        if not masks:
            return []

        depth_map = self.predict_depth(image)
        results = [self._lift_single(m, depth_map, calibration) for m in masks]
        logger.info("Lifted %d/%d masks", sum(1 for r in results if r), len(masks))
        return results
