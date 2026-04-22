"""Tests for the evaluation module — IoU, mAP, and KITTI I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autolabel3d.data.schemas import BBox3D, ObjectClass


def _make_box(cx=0.0, cy=0.0, cz=10.0, w=2.0, h=1.5, l=4.0, ry=0.0, conf=1.0, cls=ObjectClass.CAR):
    return BBox3D(
        center=np.array([cx, cy, cz], dtype=np.float64),
        dimensions=np.array([w, h, l], dtype=np.float64),
        rotation_y=ry,
        class_name=cls,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------

class TestIou3D:
    def test_identical_boxes_iou_1(self):
        from autolabel3d.evaluation.iou import compute_iou_3d
        b = _make_box()
        assert compute_iou_3d(b, b) == pytest.approx(1.0, abs=1e-6)

    def test_non_overlapping_iou_0(self):
        from autolabel3d.evaluation.iou import compute_iou_3d
        a = _make_box(cz=0.0)
        b = _make_box(cz=100.0)
        assert compute_iou_3d(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        from autolabel3d.evaluation.iou import compute_iou_3d
        a = _make_box(cx=0.0, cz=10.0, w=2.0, l=4.0, h=1.5)
        b = _make_box(cx=1.0, cz=10.0, w=2.0, l=4.0, h=1.5)
        iou = compute_iou_3d(a, b)
        assert 0.0 < iou < 1.0

    def test_no_height_overlap(self):
        from autolabel3d.evaluation.iou import compute_iou_3d
        a = _make_box(cy=0.0, h=1.0)
        b = _make_box(cy=5.0, h=1.0)  # 4m apart → no Y overlap
        assert compute_iou_3d(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        from autolabel3d.evaluation.iou import compute_iou_3d
        a = _make_box(cx=0.5)
        b = _make_box(cx=1.5)
        assert compute_iou_3d(a, b) == pytest.approx(compute_iou_3d(b, a), abs=1e-10)


class TestIouBev:
    def test_identical_bev_1(self):
        from autolabel3d.evaluation.iou import compute_iou_bev
        b = _make_box()
        assert compute_iou_bev(b, b) == pytest.approx(1.0, abs=1e-6)

    def test_non_overlapping(self):
        from autolabel3d.evaluation.iou import compute_iou_bev
        a = _make_box(cx=0.0)
        b = _make_box(cx=100.0)
        assert compute_iou_bev(a, b) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Metrics / AP
# ---------------------------------------------------------------------------

class TestComputeAp:
    def _ap(self, prec, rec):
        from autolabel3d.evaluation.metrics import _compute_ap
        return _compute_ap(np.array(prec), np.array(rec))

    def test_perfect_detector(self):
        # Precision = 1.0 for all recall levels → AP = 1.0
        prec = [1.0] * 10
        rec = [0.1 * i for i in range(1, 11)]
        ap = self._ap(prec, rec)
        assert ap == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_zero(self):
        assert self._ap([], []) == 0.0


class TestEvaluate:
    def test_perfect_predictions(self):
        from autolabel3d.evaluation.metrics import evaluate

        box = _make_box(conf=0.9)
        preds = [[box]]
        gts = [[box]]
        result = evaluate(preds, gts, iou_threshold=0.5, use_3d_iou=True)
        assert result.per_class[ObjectClass.CAR].ap == pytest.approx(1.0, abs=1e-6)

    def test_no_predictions(self):
        from autolabel3d.evaluation.metrics import evaluate

        gts = [[_make_box()]]
        result = evaluate([[]], gts, iou_threshold=0.5)
        assert result.per_class[ObjectClass.CAR].ap == pytest.approx(0.0)
        assert result.per_class[ObjectClass.CAR].num_false_negative == 1

    def test_frame_count_mismatch(self):
        from autolabel3d.evaluation.metrics import evaluate

        with pytest.raises(ValueError, match="mismatch"):
            evaluate([[_make_box()]], [[], []])

    def test_map_is_mean_of_class_aps(self):
        from autolabel3d.evaluation.metrics import evaluate

        car = _make_box(cls=ObjectClass.CAR, conf=0.9)
        ped = _make_box(cls=ObjectClass.PEDESTRIAN, conf=0.9)
        result = evaluate([[car, ped]], [[car, ped]], iou_threshold=0.5)
        expected_map = np.mean([
            result.per_class[ObjectClass.CAR].ap,
            result.per_class[ObjectClass.PEDESTRIAN].ap,
        ])
        assert result.map == pytest.approx(expected_map)


# ---------------------------------------------------------------------------
# KITTI format I/O
# ---------------------------------------------------------------------------

class TestKittiFormat:
    def test_write_and_read_roundtrip(self, tmp_path):
        from autolabel3d.evaluation.kitti_format import (
            read_kitti_annotations,
            write_kitti_annotations,
        )

        box = _make_box(cx=1.5, cy=-0.5, cz=12.0, w=1.8, h=1.5, l=4.5, ry=0.3, conf=0.85)
        out_path = tmp_path / "frame_000.txt"
        write_kitti_annotations([box], out_path)

        read_boxes = read_kitti_annotations(out_path)
        assert len(read_boxes) == 1

        rb = read_boxes[0]
        np.testing.assert_allclose(rb.center, box.center, atol=0.02)
        assert rb.width  == pytest.approx(box.width,  abs=0.02)
        assert rb.height == pytest.approx(box.height, abs=0.02)
        assert rb.length == pytest.approx(box.length, abs=0.02)
        assert rb.rotation_y == pytest.approx(box.rotation_y, abs=0.01)
        assert rb.class_name == ObjectClass.CAR

    def test_missing_file_returns_empty(self, tmp_path):
        from autolabel3d.evaluation.kitti_format import read_kitti_annotations

        boxes = read_kitti_annotations(tmp_path / "nonexistent.txt")
        assert boxes == []

    def test_empty_file_returns_empty(self, tmp_path):
        from autolabel3d.evaluation.kitti_format import (
            read_kitti_annotations,
            write_kitti_annotations,
        )

        out = tmp_path / "empty.txt"
        write_kitti_annotations([], out)
        assert read_kitti_annotations(out) == []

    def test_class_filter(self, tmp_path):
        from autolabel3d.evaluation.kitti_format import (
            read_kitti_annotations,
            write_kitti_annotations,
        )

        car_box = _make_box(cls=ObjectClass.CAR)
        ped_box = _make_box(cls=ObjectClass.PEDESTRIAN)
        out = tmp_path / "mixed.txt"
        write_kitti_annotations([car_box, ped_box], out)

        only_cars = read_kitti_annotations(out, classes={ObjectClass.CAR})
        assert len(only_cars) == 1
        assert only_cars[0].class_name == ObjectClass.CAR
