"""Tests for the segmentation module — no model weights required."""

from __future__ import annotations

import numpy as np
import pytest

from autolabel3d.data.schemas import (
    Detection2D,
    Frame,
    FrameDetections,
    FrameMasks,
    ObjectClass,
)


class TestMaskToBbox:
    def _bbox(self, mask):
        from autolabel3d.segmentation.sam2 import SAM2Segmentor
        return SAM2Segmentor._mask_to_bbox(mask)

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.bool_)
        bbox = self._bbox(mask)
        np.testing.assert_array_equal(bbox, [0, 0, 0, 0])

    def test_full_mask(self):
        mask = np.ones((10, 20), dtype=np.bool_)
        bbox = self._bbox(mask)
        # x1=0, y1=0, x2=19, y2=9
        np.testing.assert_array_equal(bbox, [0, 0, 19, 9])

    def test_rectangular_region(self):
        mask = np.zeros((100, 100), dtype=np.bool_)
        mask[20:40, 30:60] = True
        bbox = self._bbox(mask)
        np.testing.assert_array_equal(bbox, [30, 20, 59, 39])

    def test_output_dtype(self):
        mask = np.ones((5, 5), dtype=np.bool_)
        bbox = self._bbox(mask)
        assert bbox.dtype == np.float32


class TestSAM2ModelIDMapping:
    def test_known_sizes(self):
        from autolabel3d.segmentation.sam2 import SAM2_MODEL_IDS

        for size in ("tiny", "small", "base_plus", "large"):
            assert size in SAM2_MODEL_IDS
            assert "facebook/sam2" in SAM2_MODEL_IDS[size]

    def test_unknown_size_raises(self):
        from unittest.mock import MagicMock

        from autolabel3d.segmentation.sam2 import SAM2Segmentor

        cfg = MagicMock()
        cfg.model_size = "xxlarge"
        seg = SAM2Segmentor.__new__(SAM2Segmentor)
        seg.model_size = "xxlarge"
        with pytest.raises(ValueError, match="Unknown SAM 2 model size"):
            _ = seg.model_id
