"""Tests for data schemas — construction, properties, and derived values."""

from __future__ import annotations

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


class TestCameraIntrinsics:
    def test_matrix_shape(self, sample_intrinsics):
        K = sample_intrinsics.matrix
        assert K.shape == (3, 3)

    def test_matrix_values(self, sample_intrinsics):
        K = sample_intrinsics.matrix
        assert K[0, 0] == pytest.approx(sample_intrinsics.fx)
        assert K[1, 1] == pytest.approx(sample_intrinsics.fy)
        assert K[0, 2] == pytest.approx(sample_intrinsics.cx)
        assert K[1, 2] == pytest.approx(sample_intrinsics.cy)
        assert K[2, 2] == pytest.approx(1.0)


class TestFrame:
    def test_shape_properties(self, sample_frame):
        assert sample_frame.height == 480
        assert sample_frame.width == 640


class TestFrameDetections:
    def test_empty_boxes(self):
        fd = FrameDetections(frame_idx=0)
        assert fd.num_detections == 0
        assert fd.boxes.shape == (0, 4)

    def test_boxes_stacked(self, sample_detection):
        fd = FrameDetections(frame_idx=0, detections=[sample_detection])
        assert fd.num_detections == 1
        assert fd.boxes.shape == (1, 4)


class TestBBox3D:
    def test_dimension_properties(self, sample_bbox3d):
        assert sample_bbox3d.width == pytest.approx(1.8)
        assert sample_bbox3d.height == pytest.approx(1.5)
        assert sample_bbox3d.length == pytest.approx(4.5)

    def test_volume(self, sample_bbox3d):
        expected = 1.8 * 1.5 * 4.5
        assert sample_bbox3d.volume == pytest.approx(expected)

    def test_corners_shape(self, sample_bbox3d):
        corners = sample_bbox3d.corners()
        assert corners.shape == (8, 3)

    def test_corners_centered_at_box(self, sample_bbox3d):
        corners = sample_bbox3d.corners()
        centroid = corners.mean(axis=0)
        np.testing.assert_allclose(centroid, sample_bbox3d.center, atol=1e-10)


class TestFrameAnnotations:
    def test_num_annotations(self, sample_bbox3d):
        ann = FrameAnnotations(frame_idx=0, boxes_3d=[sample_bbox3d, sample_bbox3d])
        assert ann.num_annotations == 2
