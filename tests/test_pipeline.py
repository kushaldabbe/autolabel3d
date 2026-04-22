"""Tests for pipeline orchestration, factory, and CLI."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from autolabel3d.data.schemas import (
    BBox3D,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    Detection2D,
    Frame,
    FrameAnnotations,
    FrameDetections,
    FrameMasks,
    ObjectClass,
    SegmentationMask,
)
from autolabel3d.factory import (
    DETECTOR_REGISTRY,
    LIFTER_REGISTRY,
    SEGMENTOR_REGISTRY,
    build_lifter,
)
from autolabel3d.pipeline import Pipeline, PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(idx: int = 0) -> Frame:
    return Frame(
        image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        frame_idx=idx,
        timestamp=idx * 0.1,
        calibration=CameraCalibration(
            intrinsics=CameraIntrinsics(fx=1000, fy=1000, cx=320, cy=240),
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
            ),
        ),
    )


def _make_mask(frame_idx: int = 0, track_id: int = 1) -> SegmentationMask:
    mask = np.zeros((480, 640), dtype=np.bool_)
    mask[100:200, 100:200] = True
    return SegmentationMask(
        mask=mask, track_id=track_id, confidence=0.9,
        class_name=ObjectClass.CAR,
        bbox=np.array([100, 100, 200, 200], dtype=np.float32),
    )


class _MockDataLoader:
    def __len__(self): return 3
    def load_frames(self):
        for i in range(3): yield _make_frame(i)


class _MockDetector:
    def detect(self, frame, text_prompt=None):
        return FrameDetections(frame_idx=frame.frame_idx, detections=[
            Detection2D(bbox=np.array([100, 100, 200, 200], dtype=np.float32),
                        confidence=0.9, class_name=ObjectClass.CAR, class_phrase="car")
        ])
    def detect_batch(self, frames): return [self.detect(f) for f in frames]


class _MockSegmentor:
    def segment_video(self, frames, initial_detections):
        return [FrameMasks(frame_idx=f.frame_idx, masks=[_make_mask(f.frame_idx)]) for f in frames]
    def segment_frame(self, frame, detections):
        return FrameMasks(frame_idx=frame.frame_idx, masks=[_make_mask(frame.frame_idx)])


class _MockLifter:
    def lift(self, mask, calibration, image=None):
        return BBox3D(center=np.array([0.0, 1.0, 20.0]), dimensions=np.array([1.8, 1.5, 4.5]),
                      rotation_y=0.0, class_name=mask.class_name, confidence=mask.confidence,
                      track_id=mask.track_id)
    def lift_batch(self, masks, calibration, image=None):
        return [self.lift(m, calibration, image) for m in masks]


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:

    @pytest.fixture
    def pipeline(self, tmp_path):
        return Pipeline(
            data_loader=_MockDataLoader(), detector=_MockDetector(),
            segmentor=_MockSegmentor(), lifter=_MockLifter(),
            output_dir=tmp_path / "output", video_mode=True,
        )

    def test_run_video_mode(self, pipeline):
        result = pipeline.run()
        assert isinstance(result, PipelineResult)
        assert result.num_frames == 3
        assert result.num_objects == 3
        assert len(result.annotations) == 3
        assert all(isinstance(a, FrameAnnotations) for a in result.annotations)

    def test_run_frame_mode(self, tmp_path):
        pipe = Pipeline(
            data_loader=_MockDataLoader(), detector=_MockDetector(),
            segmentor=_MockSegmentor(), lifter=_MockLifter(),
            output_dir=tmp_path / "output", video_mode=False,
        )
        result = pipe.run()
        assert result.num_frames == 3
        assert result.num_objects == 3

    def test_timing_recorded(self, pipeline):
        result = pipeline.run()
        assert "total" in result.timing
        assert "detection" in result.timing
        assert "segmentation" in result.timing
        assert "lifting" in result.timing
        assert result.timing["total"] > 0

    def test_fps_positive(self, pipeline):
        result = pipeline.run()
        assert result.fps > 0

    def test_kitti_output_files(self, pipeline, tmp_path):
        pipeline.run()
        labels_dir = tmp_path / "output" / "labels"
        assert labels_dir.exists()
        assert len(list(labels_dir.glob("*.txt"))) == 3

    def test_max_frames(self, pipeline):
        result = pipeline.run(max_frames=2)
        assert result.num_frames == 2

    def test_run_with_evaluation(self, pipeline):
        gt_box = BBox3D(center=np.array([0.0, 1.0, 20.0]),
                        dimensions=np.array([1.8, 1.5, 4.5]),
                        rotation_y=0.0, class_name=ObjectClass.CAR)
        result = pipeline.run(ground_truth=[[gt_box], [gt_box], [gt_box]])
        assert result.evaluation is not None
        assert result.evaluation.map > 0

    def test_empty_dataloader(self, tmp_path):
        empty_loader = MagicMock()
        empty_loader.load_frames.return_value = iter([])
        pipe = Pipeline(
            data_loader=empty_loader, detector=_MockDetector(),
            segmentor=_MockSegmentor(), lifter=_MockLifter(),
            output_dir=tmp_path / "output",
        )
        result = pipe.run()
        assert result.num_frames == 0
        assert result.num_objects == 0

    def test_no_detections_first_frame(self, tmp_path):
        empty_detector = MagicMock()
        empty_detector.detect.return_value = FrameDetections(frame_idx=0, detections=[])
        pipe = Pipeline(
            data_loader=_MockDataLoader(), detector=empty_detector,
            segmentor=_MockSegmentor(), lifter=_MockLifter(),
            output_dir=tmp_path / "output", video_mode=True,
        )
        result = pipe.run()
        assert result.num_frames == 3
        assert result.num_objects == 0

    def test_no_calibration_skips_lifting(self, tmp_path):
        class NoCaliLoader:
            def __len__(self): return 1
            def load_frames(self):
                yield Frame(image=np.zeros((480, 640, 3), dtype=np.uint8),
                            frame_idx=0, timestamp=0.0, calibration=None)

        pipe = Pipeline(
            data_loader=NoCaliLoader(), detector=_MockDetector(),
            segmentor=_MockSegmentor(), lifter=_MockLifter(), video_mode=False,
        )
        result = pipe.run()
        assert result.num_objects == 0

    def test_summary_string(self, pipeline):
        result = pipeline.run()
        summary = result.summary()
        assert "FPS" in summary
        assert "total" in summary


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:

    def test_registries_not_empty(self):
        assert len(DETECTOR_REGISTRY) >= 1
        assert len(SEGMENTOR_REGISTRY) >= 1
        assert len(LIFTER_REGISTRY) >= 2

    def test_build_boxer_lifter(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            name="boxer",
            class_dimensions=SimpleNamespace(
                car=SimpleNamespace(width=1.8, height=1.5, length=4.5),
                pedestrian=SimpleNamespace(width=0.6, height=1.7, length=0.6),
                cyclist=SimpleNamespace(width=0.6, height=1.7, length=1.8),
                traffic_cone=SimpleNamespace(width=0.3, height=0.8, length=0.3),
            ),
        )
        lifter = build_lifter(cfg)
        from autolabel3d.lifting.boxer import BoxerLifter
        assert isinstance(lifter, BoxerLifter)

    def test_unknown_lifter_raises(self):
        cfg = SimpleNamespace(name="nonexistent_lifter")
        with pytest.raises(ValueError, match="Unknown lifter"):
            build_lifter(cfg)
