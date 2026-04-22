"""ONNX Runtime inference session for Depth Anything V2.

Drop-in replacement for the PyTorch model used inside DepthAnythingLifter.

ONNX Runtime Execution Providers (ranked by speed):
    TensorRT > CoreML > CUDA > CPU

ORT applies graph-level optimizations automatically:
    - Operator fusion (Conv + BN + ReLU → single fused kernel)
    - Constant folding (evaluate static subgraphs at load time)
    - Memory planning (shared buffer allocation, in-place ops)
    - Shape inference (enables specialized kernels for known shapes)

Typical speedup vs PyTorch eager: 1.5–3× even on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)


class OnnxDepthEstimator:
    """ONNX Runtime wrapper for Depth Anything V2.

    Handles preprocessing (resize + normalize) and postprocessing
    (resize to original resolution + reciprocal inversion).
    """

    #: ImageNet normalization (same as HuggingFace AutoImageProcessor)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        """Initialize an ONNX Runtime inference session.

        Args:
            model_path: Path to the .onnx file.
            providers: Execution providers. None = auto-detect best available.
        """
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        if providers is None:
            providers = self._select_providers(ort.get_available_providers())

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # use all cores
        sess_options.inter_op_num_threads = 0

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

        meta_in  = self.session.get_inputs()[0]
        meta_out = self.session.get_outputs()[0]
        self.input_name  = meta_in.name
        self.output_name = meta_out.name
        shape = meta_in.shape
        self._input_h = shape[2] if isinstance(shape[2], int) else 518
        self._input_w = shape[3] if isinstance(shape[3], int) else 518

        logger.info(
            "ONNX session ready: %s  input=%dx%d  provider=%s",
            self.model_path.name, self._input_h, self._input_w,
            self.session.get_providers()[0],
        )

    @staticmethod
    def _select_providers(available: list[str]) -> list[str]:
        """Pick the best providers from what ORT reports as available."""
        selected: list[str] = []
        for ep in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
            if ep in available:
                selected.append(ep)
        selected.append("CPUExecutionProvider")
        return selected

    def preprocess(self, image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Preprocess image for ONNX inference.

        Replicates HuggingFace AutoImageProcessor:
            1. BGR → RGB
            2. Resize to model input size (bilinear)
            3. Normalize: (pixel/255 − mean) / std
            4. HWC → NCHW (add batch dim)

        Returns:
            (1, 3, H_model, W_model) float32 array.
        """
        import cv2

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)

    def predict(self, image_bgr: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Run depth estimation on a single BGR image.

        Returns:
            (H, W) depth map at original resolution. Higher value = farther.
        """
        import cv2

        original_h, original_w = image_bgr.shape[:2]
        input_tensor = self.preprocess(image_bgr)

        (depth_pred,) = self.session.run([self.output_name], {self.input_name: input_tensor})
        depth_pred = depth_pred.squeeze().astype(np.float64)  # (H_model, W_model)

        depth_resized = cv2.resize(depth_pred, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Reciprocal inversion: Depth Anything outputs inverse depth
        return 1.0 / (depth_resized + 1e-6)
