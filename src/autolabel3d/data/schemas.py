"""Core data schemas for the autolabel3d pipeline.

These dataclasses define the data structures that flow between pipeline stages.
They are the "language" all modules speak — detection outputs become segmentation
inputs, which become lifting inputs, and so on.

Design decisions:
    - Python dataclasses (not Pydantic) for simplicity and PyTorch compatibility.
    - numpy arrays for numeric data — interops with both PyTorch and OpenCV.
    - Immutable where sensible (frozen=True on value objects like calibration).
    - Each schema maps to a real concept in the AV perception stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObjectClass(str, Enum):
    """Object classes detected and labeled by the pipeline.

    Using str enum so values serialize cleanly to JSON/YAML and match
    both nuScenes category names and KITTI class strings.
    """

    CAR = "car"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_CONE = "traffic_cone"


# ---------------------------------------------------------------------------
# Camera Calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters — pinhole model.

    The intrinsic matrix K projects 3D camera-frame points to 2D pixels:

        [u]     [fx  0  cx] [X]
        [v]  =  [ 0 fy  cy] [Y] / Z
        [1]     [ 0  0   1] [Z]

    Attributes:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point (optical axis intersection with image plane).
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> NDArray[np.float64]:
        """3×3 intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera extrinsic parameters — pose in world/ego frame.

    Defines the rigid transform from camera frame to world/ego frame:
        - rotation:    3×3 rotation matrix (camera → world)
        - translation: 3-vector (camera origin in world coordinates)
    """

    rotation: NDArray[np.float64]     # (3, 3)
    translation: NDArray[np.float64]  # (3,)

    @property
    def transform_matrix(self) -> NDArray[np.float64]:
        """4×4 homogeneous transform matrix (camera → world)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass(frozen=True)
class CameraCalibration:
    """Complete camera calibration (intrinsics + extrinsics)."""

    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """A single camera frame flowing through the pipeline.

    Carries the BGR image (OpenCV convention), metadata, and calibration
    needed for 3D lifting.
    """

    image: NDArray[np.uint8]              # (H, W, 3) BGR
    frame_idx: int                         # sequential index in the scene/video
    timestamp: float                       # seconds
    camera_name: str = "CAM_FRONT"
    calibration: CameraCalibration | None = None
    source_path: Path | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


# ---------------------------------------------------------------------------
# 2D Detection — output of Grounding DINO
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection2D:
    """A single 2D object detection from Grounding DINO.

    bbox is in [x1, y1, x2, y2] pixel format (top-left, bottom-right).
    """

    bbox: NDArray[np.float32]   # (4,) [x1, y1, x2, y2]
    confidence: float            # detector score [0, 1]
    class_name: ObjectClass
    class_phrase: str = ""       # raw matched text phrase from the model


@dataclass
class FrameDetections:
    """All 2D detections for a single frame."""

    frame_idx: int
    detections: list[Detection2D] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def boxes(self) -> NDArray[np.float32]:
        """All detection boxes as (N, 4) array."""
        if not self.detections:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack([d.bbox for d in self.detections])


# ---------------------------------------------------------------------------
# Segmentation Mask — output of SAM 2
# ---------------------------------------------------------------------------

@dataclass
class SegmentationMask:
    """Per-object pixel mask for a single frame from SAM 2.

    track_id links the same physical object across frames (temporal tracking).
    """

    mask: NDArray[np.bool_]      # (H, W) binary mask; True = object pixel
    track_id: int                 # consistent across frames in a clip
    confidence: float             # SAM 2 IoU quality score
    class_name: ObjectClass
    bbox: NDArray[np.float32]    # (4,) tight bbox [x1, y1, x2, y2] around the mask


@dataclass
class FrameMasks:
    """All segmentation masks for a single frame."""

    frame_idx: int
    masks: list[SegmentationMask] = field(default_factory=list)

    @property
    def num_masks(self) -> int:
        return len(self.masks)


# ---------------------------------------------------------------------------
# 3D Bounding Box — final pipeline output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox3D:
    """A 3D bounding box in world/ego coordinates (KITTI convention).

    Attributes:
        center:     (3,) [x, y, z] in meters. KITTI: X=right, Y=down, Z=forward.
        dimensions: (3,) [width, height, length] in meters.
        rotation_y: Yaw angle around Y-axis in radians.
        class_name: Object class label.
        confidence: Prediction confidence [0, 1].
        track_id:   Links to the corresponding SAM 2 track (-1 if untracked).
    """

    center: NDArray[np.float64]       # (3,) [x, y, z]
    dimensions: NDArray[np.float64]   # (3,) [width, height, length]
    rotation_y: float
    class_name: ObjectClass
    confidence: float = 1.0
    track_id: int = -1

    @property
    def width(self) -> float:
        return float(self.dimensions[0])

    @property
    def height(self) -> float:
        return float(self.dimensions[1])

    @property
    def length(self) -> float:
        return float(self.dimensions[2])

    @property
    def volume(self) -> float:
        return float(np.prod(self.dimensions))

    def corners(self) -> NDArray[np.float64]:
        """Compute 8 corners of the 3D box in world frame.

        Returns (8, 3) array, ordered for wireframe rendering.
        Rotates a unit box by rotation_y around Y, then translates to center.
        """
        w, h, l = self.dimensions
        x = np.array([ w,  w, -w, -w,  w,  w, -w, -w]) / 2
        y = np.array([-h, -h, -h, -h,  h,  h,  h,  h]) / 2
        z = np.array([ l, -l, -l,  l,  l, -l, -l,  l]) / 2

        cos_r = np.cos(self.rotation_y)
        sin_r = np.sin(self.rotation_y)
        R = np.array([
            [cos_r, 0, sin_r],
            [0,     1, 0    ],
            [-sin_r, 0, cos_r],
        ])

        corners = np.stack([x, y, z], axis=1)  # (8, 3)
        return corners @ R.T + self.center


@dataclass
class FrameAnnotations:
    """All 3D annotations for a single frame — the pipeline's final output."""

    frame_idx: int
    boxes_3d: list[BBox3D] = field(default_factory=list)
    detections: FrameDetections | None = None
    masks: FrameMasks | None = None

    @property
    def num_annotations(self) -> int:
        return len(self.boxes_3d)
