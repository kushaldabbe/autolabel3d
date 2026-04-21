"""Abstract base class for 2D-to-3D lifting modules.

Lifting recovers metric 3D bounding boxes from 2D segmentation masks and
camera calibration. Two strategies are provided:
    - Boxer: geometric constraints + class size priors (no neural network)
    - DepthAnythingLifter: monocular depth estimation via Depth Anything V2
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from autolabel3d.data.schemas import BBox3D, CameraCalibration, SegmentationMask


class BaseLifter(ABC):
    """Abstract 2D-to-3D lifting module."""

    @abstractmethod
    def lift(
        self,
        mask: SegmentationMask,
        calibration: CameraCalibration,
        image: np.ndarray | None = None,
    ) -> BBox3D | None:
        """Lift a single 2D mask to a 3D bounding box.

        Args:
            mask: 2D segmentation mask with bounding box.
            calibration: Camera intrinsics + extrinsics.
            image: Original frame image (required by depth-based lifters).

        Returns:
            3D bounding box, or None if lifting fails.
        """
        ...

    @abstractmethod
    def lift_batch(
        self,
        masks: list[SegmentationMask],
        calibration: CameraCalibration,
        image: np.ndarray | None = None,
    ) -> list[BBox3D | None]:
        """Lift multiple masks from a single frame to 3D boxes."""
        ...
