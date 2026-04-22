"""Tests for data loaders (dashcam) — no model weights required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from autolabel3d.data.dashcam_loader import DashcamLoader


def _make_dashcam_cfg(video_path: Path, every_n: int = 1, max_frames: int = 5):
    return OmegaConf.create({
        "name": "dashcam",
        "video_path": str(video_path),
        "sampling": {
            "strategy": "every_n",
            "every_n": every_n,
            "max_frames": max_frames,
        },
        "classes": ["car", "pedestrian"],
        "calibration": {
            "fx": 500.0, "fy": 500.0,
            "cx": 160.0, "cy": 120.0,
        },
    })


class TestDashcamLoader:
    def test_len(self, tmp_video):
        cfg = _make_dashcam_cfg(tmp_video, every_n=1, max_frames=5)
        loader = DashcamLoader(cfg)
        assert len(loader) == 5

    def test_load_frames_yields_correct_count(self, tmp_video):
        cfg = _make_dashcam_cfg(tmp_video, every_n=2, max_frames=4)
        loader = DashcamLoader(cfg)
        frames = list(loader.load_frames())
        assert len(frames) == len(loader)

    def test_frame_has_calibration(self, tmp_video):
        cfg = _make_dashcam_cfg(tmp_video)
        loader = DashcamLoader(cfg)
        frame = loader.get_frame(0)
        assert frame.calibration is not None
        assert frame.calibration.intrinsics.fx == pytest.approx(500.0)

    def test_frame_image_shape(self, tmp_video):
        cfg = _make_dashcam_cfg(tmp_video)
        loader = DashcamLoader(cfg)
        frame = loader.get_frame(0)
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3

    def test_get_frame_out_of_range(self, tmp_video):
        cfg = _make_dashcam_cfg(tmp_video, max_frames=3)
        loader = DashcamLoader(cfg)
        with pytest.raises(IndexError):
            loader.get_frame(999)

    def test_missing_video_raises(self, tmp_path):
        cfg = _make_dashcam_cfg(tmp_path / "nonexistent.mp4")
        with pytest.raises(FileNotFoundError):
            DashcamLoader(cfg)
