"""Tests for the 3D lifting module — no model weights required."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from autolabel3d.data.schemas import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    ObjectClass,
    SegmentationMask,
)
from autolabel3d.utils.geometry import (
    depth_map_to_pointcloud,
    fit_3d_bbox_pca,
    pixel_to_ray,
    transform_points,
)


def _make_calibration(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    intr = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    extr = CameraExtrinsics(rotation=np.eye(3), translation=np.zeros(3))
    return CameraCalibration(intrinsics=intr, extrinsics=extr)


def _make_mask(h=480, w=640, row_slice=slice(100, 200), col_slice=slice(200, 400)):
    mask_arr = np.zeros((h, w), dtype=np.bool_)
    mask_arr[row_slice, col_slice] = True
    bbox = np.array([col_slice.start, row_slice.start, col_slice.stop, row_slice.stop], dtype=np.float32)
    return SegmentationMask(
        mask=mask_arr,
        track_id=1,
        confidence=0.9,
        class_name=ObjectClass.CAR,
        bbox=bbox,
    )


# ------------------------------------------------------------------
# Geometry utilities
# ------------------------------------------------------------------

class TestPixelToRay:
    def test_principal_point_is_z_axis(self):
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        rays = pixel_to_ray(
            np.array([320.0]), np.array([240.0]), intr
        )
        np.testing.assert_allclose(rays[0], [0, 0, 1], atol=1e-10)

    def test_output_unit_length(self):
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        us = np.array([0.0, 100.0, 320.0, 640.0])
        vs = np.array([0.0, 100.0, 240.0, 480.0])
        rays = pixel_to_ray(us, vs, intr)
        norms = np.linalg.norm(rays, axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-10)


class TestDepthMapToPointcloud:
    def test_output_shape(self):
        depth = np.ones((480, 640), dtype=np.float64) * 5.0
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pts = depth_map_to_pointcloud(depth, intr)
        assert pts.shape[1] == 3
        assert len(pts) == 480 * 640

    def test_mask_filters_points(self):
        depth = np.ones((480, 640), dtype=np.float64) * 5.0
        mask = np.zeros((480, 640), dtype=np.bool_)
        mask[100:200, 100:200] = True
        intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        pts = depth_map_to_pointcloud(depth, intr, mask=mask)
        assert len(pts) == 100 * 100

    def test_zero_depth_filtered(self):
        depth = np.zeros((10, 10), dtype=np.float64)
        intr = CameraIntrinsics(fx=50.0, fy=50.0, cx=5.0, cy=5.0)
        pts = depth_map_to_pointcloud(depth, intr)
        assert len(pts) == 0


class TestTransformPoints:
    def test_identity(self):
        pts = np.random.randn(50, 3)
        out = transform_points(pts, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_translation(self):
        pts = np.zeros((5, 3))
        out = transform_points(pts, np.eye(3), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, np.tile([1, 2, 3], (5, 1)), atol=1e-12)


class TestFit3dBboxPca:
    def test_axis_aligned_box(self):
        # A box of known dimensions aligned with camera axes
        rng = np.random.default_rng(42)
        pts = rng.uniform(low=[0, 0, 0], high=[2, 1, 5], size=(500, 3))
        center, dims, rot_y = fit_3d_bbox_pca(pts)

        # Centroid should be near (1, 0.5, 2.5)
        np.testing.assert_allclose(center, [1.0, 0.5, 2.5], atol=0.1)
        # Volume should be preserved
        assert dims[0] * dims[1] * dims[2] == pytest.approx(2 * 1 * 5, rel=0.1)

    def test_degenerate_few_points(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]])
        center, dims, rot_y = fit_3d_bbox_pca(pts)
        assert dims.shape == (3,)


# ------------------------------------------------------------------
# Boxer lifter
# ------------------------------------------------------------------

class TestBoxerLifter:
    def _make_boxer(self):
        cfg = OmegaConf.create({
            "fitting": {
                "camera_height": 1.65,
                "use_ground_plane": True,
            }
        })
        from autolabel3d.lifting.boxer import BoxerLifter
        return BoxerLifter(cfg)

    def test_returns_bbox3d(self, sample_mask):
        boxer = self._make_boxer()
        cal = _make_calibration()
        result = boxer.lift(sample_mask, cal)
        assert result is not None
        assert result.class_name == ObjectClass.CAR

    def test_distance_is_positive(self, sample_mask):
        boxer = self._make_boxer()
        cal = _make_calibration()
        result = boxer.lift(sample_mask, cal)
        assert result.center[2] > 0  # positive depth

    def test_lift_batch_length(self, sample_mask):
        boxer = self._make_boxer()
        cal = _make_calibration()
        masks = [sample_mask, sample_mask]
        results = boxer.lift_batch(masks, cal)
        assert len(results) == 2

    def test_triangle_similarity_distance(self):
        from autolabel3d.lifting.boxer import BoxerLifter
        boxer = BoxerLifter.__new__(BoxerLifter)
        boxer.camera_height = 1.65
        boxer.use_ground_plane = True
        # 2D box height = 100px, fy = 500, h_prior = 1.5m → Z = 7.5m
        bbox = np.array([0.0, 100.0, 200.0, 200.0], dtype=np.float32)
        Z = boxer._estimate_distance(bbox, fy=500.0, h_prior=1.5)
        assert Z == pytest.approx(7.5, rel=1e-6)
