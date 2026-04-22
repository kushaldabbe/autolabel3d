"""Dashcam video data loader.

Loads frames from an MP4/AVI/MOV file using OpenCV. Simpler than nuScenes —
no ground truth annotations, and camera calibration is supplied via config.

Useful for:
    - Running the pipeline on your own dashcam or GoPro footage
    - Demo / visualization on arbitrary video
    - Quick end-to-end testing without a large dataset
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from autolabel3d.data.base import BaseDataLoader
from autolabel3d.data.schemas import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    Frame,
)
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)


class DashcamLoader(BaseDataLoader):
    """Loads sampled frames from a dashcam video file.

    Example:
        loader = DashcamLoader(cfg)          # cfg from configs/data/dashcam.yaml
        for frame in loader.load_frames():
            process(frame.image)             # (H, W, 3) BGR numpy array
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.video_path = Path(cfg.video_path)
        self.every_n: int = cfg.sampling.every_n
        self.max_frames: int | None = cfg.sampling.get("max_frames", None)

        self._calibration = self._build_calibration()
        self._total_frames: int = 0
        self._fps: float = 0.0
        self._frame_indices: list[int] = []
        self._probe_video()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_calibration(self) -> CameraCalibration:
        """Build CameraCalibration from config values.

        Identity extrinsics: camera frame is treated as world frame.
        """
        c = self.cfg.calibration
        return CameraCalibration(
            intrinsics=CameraIntrinsics(
                fx=float(c.fx), fy=float(c.fy),
                cx=float(c.cx), cy=float(c.cy),
            ),
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
            ),
        )

    def _probe_video(self) -> None:
        """Read video metadata and build the frame-index list to process."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self._frame_indices = list(range(0, self._total_frames, self.every_n))
        if self.max_frames:
            self._frame_indices = self._frame_indices[: self.max_frames]

        logger.info(
            "Video: %s | %d total frames @ %.1f FPS | "
            "sampling every %d → %d frames to process",
            self.video_path.name, self._total_frames, self._fps,
            self.every_n, len(self._frame_indices),
        )

    # ------------------------------------------------------------------
    # BaseDataLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._frame_indices)

    def load_frames(self) -> Iterator[Frame]:
        """Yield sampled frames sequentially, keeping only one open handle."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        try:
            for output_idx, video_frame_idx in enumerate(self._frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, image = cap.read()
                if not ret:
                    logger.warning("Failed to read frame %d, skipping", video_frame_idx)
                    continue

                yield Frame(
                    image=image,
                    frame_idx=output_idx,
                    timestamp=video_frame_idx / self._fps,
                    camera_name="dashcam",
                    calibration=self._calibration,
                    source_path=self.video_path,
                )
        finally:
            cap.release()

    def get_frame(self, idx: int) -> Frame:
        """Load a specific sampled frame by output index."""
        if idx < 0 or idx >= len(self._frame_indices):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self._frame_indices)})")

        video_frame_idx = self._frame_indices[idx]
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ret, image = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {video_frame_idx}")
        finally:
            cap.release()

        return Frame(
            image=image,
            frame_idx=idx,
            timestamp=video_frame_idx / self._fps,
            camera_name="dashcam",
            calibration=self._calibration,
            source_path=self.video_path,
        )
