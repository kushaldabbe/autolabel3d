"""Tests for the detection module — no model weights required."""

from __future__ import annotations

import numpy as np
import pytest

from autolabel3d.data.schemas import ObjectClass


class TestComputeIou1VsN:
    """Unit tests for the NMS IoU kernel — no model needed."""

    def _iou(self, box, boxes):
        from autolabel3d.detection.grounding_dino import GroundingDINODetector
        return GroundingDINODetector._compute_iou_1_vs_n(
            np.array(box, dtype=float),
            np.array(boxes, dtype=float),
        )

    def test_identical_boxes(self):
        iou = self._iou([0, 0, 10, 10], [[0, 0, 10, 10]])
        assert iou[0] == pytest.approx(1.0)

    def test_no_overlap(self):
        iou = self._iou([0, 0, 5, 5], [[10, 10, 20, 20]])
        assert iou[0] == pytest.approx(0.0)

    def test_half_overlap(self):
        # Box A: [0,0,10,10], Box B: [5,0,15,10]
        # Intersection: [5,0,10,10] = 50, Union: 100+100-50 = 150
        iou = self._iou([0, 0, 10, 10], [[5, 0, 15, 10]])
        assert iou[0] == pytest.approx(50 / 150)

    def test_multiple_boxes(self):
        iou = self._iou(
            [0, 0, 10, 10],
            [[0, 0, 10, 10], [10, 10, 20, 20], [5, 5, 15, 15]],
        )
        assert iou[0] == pytest.approx(1.0)
        assert iou[1] == pytest.approx(0.0)
        assert iou[2] > 0.0


class TestMapPhraseToClass:
    def _map(self, phrase):
        from autolabel3d.detection.grounding_dino import GroundingDINODetector
        return GroundingDINODetector._map_phrase_to_class(phrase)

    def test_exact_match_car(self):
        assert self._map("car") == ObjectClass.CAR

    def test_exact_match_pedestrian(self):
        assert self._map("pedestrian") == ObjectClass.PEDESTRIAN

    def test_substring_match(self):
        assert self._map("a white car") == ObjectClass.CAR

    def test_unknown_phrase(self):
        assert self._map("airplane") is None

    def test_case_insensitive_after_lower(self):
        # _map_phrase_to_class receives lowercased string from _parse_results
        assert self._map("traffic cone") == ObjectClass.TRAFFIC_CONE


class TestNMSLogic:
    """Test the NMS algorithm without loading the full model."""

    def _run_nms(self, boxes_list, scores, classes, iou_thresh=0.5):
        from unittest.mock import MagicMock

        from autolabel3d.data.schemas import Detection2D, FrameDetections
        from autolabel3d.detection.grounding_dino import GroundingDINODetector

        cfg = MagicMock()
        cfg.box_threshold = 0.3
        cfg.text_threshold = 0.25
        cfg.nms.enabled = True
        cfg.nms.iou_threshold = iou_thresh
        cfg.get.return_value = None

        detector = GroundingDINODetector.__new__(GroundingDINODetector)
        detector.cfg = cfg
        detector.nms_iou_threshold = iou_thresh

        detections = FrameDetections(
            frame_idx=0,
            detections=[
                Detection2D(
                    bbox=np.array(b, dtype=np.float32),
                    confidence=s,
                    class_name=c,
                    class_phrase=c.value,
                )
                for b, s, c in zip(boxes_list, scores, classes)
            ],
        )
        return detector._apply_nms(detections)

    def test_nms_removes_duplicate(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
        scores = [0.9, 0.8]
        classes = [ObjectClass.CAR, ObjectClass.CAR]
        result = self._run_nms(boxes, scores, classes, iou_thresh=0.5)
        assert result.num_detections == 1
        assert result.detections[0].confidence == pytest.approx(0.9)

    def test_nms_keeps_non_overlapping(self):
        boxes = [[0, 0, 10, 10], [100, 100, 110, 110]]
        scores = [0.9, 0.8]
        classes = [ObjectClass.CAR, ObjectClass.CAR]
        result = self._run_nms(boxes, scores, classes, iou_thresh=0.5)
        assert result.num_detections == 2

    def test_nms_per_class_no_cross_suppression(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        scores = [0.9, 0.8]
        classes = [ObjectClass.CAR, ObjectClass.PEDESTRIAN]
        result = self._run_nms(boxes, scores, classes, iou_thresh=0.5)
        assert result.num_detections == 2
