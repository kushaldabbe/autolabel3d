"""Shared pytest fixtures for the autolabel3d test suite."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from autolabel3d.data.schemas import (
    BBox3D,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    Detection2D,
    Frame,
    ObjectClass,
    SegmentationMask,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Small synthetic BGR image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=1000.0, fy=1000.0, cx=320.0, cy=240.0)


@pytest.fixture
def sample_extrinsics() -> CameraExtrinsics:
    return CameraExtrinsics(
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
    )


@pytest.fixture
def sample_calibration(sample_intrinsics, sample_extrinsics) -> CameraCalibration:
    return CameraCalibration(intrinsics=sample_intrinsics, extrinsics=sample_extrinsics)


@pytest.fixture
def sample_frame(sample_image, sample_calibration) -> Frame:
    return Frame(
        image=sample_image,
        frame_idx=0,
        timestamp=0.0,
        camera_name="CAM_FRONT",
        calibration=sample_calibration,
    )


@pytest.fixture
def sample_detection() -> Detection2D:
    return Detection2D(
        bbox=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        confidence=0.9,
        class_name=ObjectClass.CAR,
        class_phrase="car",
    )


@pytest.fixture
def sample_mask() -> SegmentationMask:
    mask = np.zeros((480, 640), dtype=np.bool_)
    mask[100:200, 100:200] = True
    return SegmentationMask(
        mask=mask,
        track_id=1,
        confidence=0.95,
        class_name=ObjectClass.CAR,
        bbox=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
    )


@pytest.fixture
def sample_bbox3d() -> BBox3D:
    return BBox3D(
        center=np.array([5.0, 1.0, 20.0], dtype=np.float64),
        dimensions=np.array([1.8, 1.5, 4.5], dtype=np.float64),
        rotation_y=0.0,
        class_name=ObjectClass.CAR,
    )


@pytest.fixture
def tmp_video(tmp_path) -> Path:
    """Create a small synthetic MP4 video for testing the dashcam loader."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for i in range(15):
        frame = np.full((240, 320, 3), i * 17, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path
