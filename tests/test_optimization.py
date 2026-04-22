"""Tests for the ONNX optimization module.

Covers:
    - ONNX export (mocked model, no weight download)
    - OnnxDepthEstimator preprocessing (exact ImageNet normalization)
    - Execution provider selection logic
    - End-to-end predict with a tiny synthetic ONNX model
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportDepthAnything:

    def test_export_creates_onnx_file(self, tmp_path):
        import torch

        class TinyDepthModel(torch.nn.Module):
            def forward(self, pixel_values):
                return pixel_values[:, :1, :, :]  # (1, 1, H, W) → treated as (1, H, W)

        with patch("transformers.AutoModelForDepthEstimation") as mock_cls:
            mock_cls.from_pretrained.return_value = TinyDepthModel()
            from autolabel3d.optimization.export import export_depth_anything
            result = export_depth_anything(
                model_size="small",
                output_path=tmp_path / "test.onnx",
                simplify=False,
            )

        assert result.exists()
        assert result.stat().st_size > 0
        assert result.suffix == ".onnx"

    def test_export_creates_parent_dirs(self, tmp_path):
        import torch

        class TinyModel(torch.nn.Module):
            def forward(self, pixel_values):
                return pixel_values[:, :1, :, :]

        with patch("transformers.AutoModelForDepthEstimation") as mock_cls:
            mock_cls.from_pretrained.return_value = TinyModel()
            from autolabel3d.optimization.export import export_depth_anything
            result = export_depth_anything(
                model_size="small",
                output_path=tmp_path / "nested" / "dir" / "model.onnx",
                simplify=False,
            )

        assert result.parent.exists()

    def test_validate_onnx_passes(self, tmp_path):
        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        graph = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "g", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = tmp_path / "valid.onnx"
        onnx.save(model, str(path))

        from autolabel3d.optimization.export import _validate_onnx
        _validate_onnx(path)  # should not raise

    def test_log_model_stats_no_error(self, tmp_path):
        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        graph = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "g", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = tmp_path / "stats.onnx"
        onnx.save(model, str(path))

        from autolabel3d.optimization.export import _log_model_stats
        _log_model_stats(path)  # should not raise


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestOnnxPreprocessing:

    def _make_stub(self):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        stub = object.__new__(OnnxDepthEstimator)
        stub.model_path = Path("fake.onnx")
        stub.session = None
        stub.input_name = "pixel_values"
        stub.output_name = "predicted_depth"
        stub._input_h = 518
        stub._input_w = 518
        return stub

    def test_preprocess_output_shape(self, sample_image):
        stub = self._make_stub()
        result = stub.preprocess(sample_image)
        assert result.shape == (1, 3, 518, 518)
        assert result.dtype == np.float32

    def test_preprocess_imagenet_normalization(self):
        stub = self._make_stub()
        image = np.full((100, 100, 3), 127, dtype=np.uint8)
        result = stub.preprocess(image)

        pixel_val = 127.0 / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        for c in range(3):
            expected = (pixel_val - mean[c]) / std[c]
            np.testing.assert_allclose(result[0, c].mean(), expected, atol=0.02)

    def test_preprocess_bgr_to_rgb(self):
        stub = self._make_stub()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 2] = 255  # BGR: channel 2 = Red
        result = stub.preprocess(image)
        # After BGR→RGB, channel 0 = Red (high), channel 2 = Blue (low)
        assert result[0, 0].mean() > result[0, 2].mean()


# ---------------------------------------------------------------------------
# Provider selection tests
# ---------------------------------------------------------------------------


class TestProviderSelection:

    def test_coreml_preferred(self):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        providers = OnnxDepthEstimator._select_providers(
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )
        assert providers[0] == "CoreMLExecutionProvider"
        assert providers[-1] == "CPUExecutionProvider"

    def test_cuda_when_no_coreml(self):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        providers = OnnxDepthEstimator._select_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        assert providers[0] == "CUDAExecutionProvider"

    def test_cpu_only_fallback(self):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        providers = OnnxDepthEstimator._select_providers(["CPUExecutionProvider"])
        assert providers == ["CPUExecutionProvider"]

    def test_priority_order_with_all_providers(self):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        providers = OnnxDepthEstimator._select_providers(
            ["CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"]
        )
        assert providers == ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# End-to-end with tiny ONNX model
# ---------------------------------------------------------------------------


class TestOnnxEndToEnd:

    @pytest.fixture
    def tiny_model(self, tmp_path) -> Path:
        """Tiny ONNX model: (1,3,518,518) → (1,518,518) via ReduceMean."""
        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 518, 518])
        Y = helper.make_tensor_value_info("predicted_depth", TensorProto.FLOAT, [1, 518, 518])
        node = helper.make_node("ReduceMean", ["pixel_values"], ["predicted_depth"],
                                axes=[1], keepdims=0)
        graph = helper.make_graph([node], "tiny_depth", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        path = tmp_path / "tiny.onnx"
        onnx.save(model, str(path))
        return path

    def test_predict_shape(self, tiny_model, sample_image):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        est = OnnxDepthEstimator(tiny_model, providers=["CPUExecutionProvider"])
        depth = est.predict(sample_image)
        assert depth.shape == (sample_image.shape[0], sample_image.shape[1])

    def test_predict_finite_values(self, tiny_model, sample_image):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        est = OnnxDepthEstimator(tiny_model, providers=["CPUExecutionProvider"])
        depth = est.predict(sample_image)
        assert np.all(np.isfinite(depth))

    def test_missing_model_raises(self):
        from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            OnnxDepthEstimator("nonexistent.onnx")
