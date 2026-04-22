"""ONNX export for Depth Anything V2.

ONNX (Open Neural Network Exchange) is a vendor-neutral format for ML models.
Exporting to ONNX unlocks:
    - ONNX Runtime (ORT) — faster CPU/GPU inference with graph optimizations
    - CoreML — native Apple Silicon via the Neural Engine
    - TensorRT — NVIDIA int8/fp16 GPU kernels
    - OpenVINO — Intel CPU/VPU optimization

Why depth model only?
    - Grounding DINO has BERT text encoding + deformable attention (custom ops)
    - SAM 2 has stateful memory + control flow (not ONNX-friendly)
    - Depth Anything V2 is a standard ViT encoder-decoder — clean export

Why ONNX over torch.compile?
    - ONNX Runtime has hand-tuned CoreML delegation on Apple Silicon
    - Portable: same .onnx runs on any platform/runtime
    - In production AV stacks, ONNX/TensorRT is the standard deployment format
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

# Standard ViT patch size is 14px → input must be a multiple of 14
DEFAULT_INPUT_HEIGHT = 518
DEFAULT_INPUT_WIDTH = 518


def export_depth_anything(
    model_size: str = "small",
    output_path: str | Path = "weights/depth_anything_v2.onnx",
    input_height: int = DEFAULT_INPUT_HEIGHT,
    input_width: int = DEFAULT_INPUT_WIDTH,
    opset_version: int = 17,
    simplify: bool = True,
) -> Path:
    """Export Depth Anything V2 to ONNX format.

    Key export decisions:

    - opset_version=17: minimum for reliable ViT export (Oct 2022); supports
      all ops in the DPT decoder (resize, layer_norm).
    - Fixed spatial dims: ViT positional embeddings are resolution-dependent;
      fixed shapes also enable aggressive ORT graph optimization.
    - simplify=True: constant folding + dead code elimination, typically
      reduces node count by 10-30%.

    Args:
        model_size: "small", "base", or "large".
        output_path: Destination .onnx file.
        input_height: Fixed input height (multiple of 14 for ViT patches).
        input_width: Fixed input width.
        opset_version: ONNX opset.
        simplify: Whether to run onnx-simplifier after export.

    Returns:
        Path to the exported ONNX file.
    """
    from transformers import AutoModelForDepthEstimation

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    logger.info("Loading %s for ONNX export...", model_id)

    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    model.cpu()  # export on CPU for broadest compatibility

    dummy_input = torch.randn(1, 3, input_height, input_width)

    logger.info(
        "Exporting to ONNX (opset=%d, input=%dx%d)...",
        opset_version, input_height, input_width,
    )

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        opset_version=opset_version,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values":    {0: "batch_size"},
            "predicted_depth": {0: "batch_size"},
        },
    )

    logger.info("Exported to %s", output_path)
    _validate_onnx(output_path)

    if simplify:
        _simplify_onnx(output_path)

    _log_model_stats(output_path)
    return output_path


def _validate_onnx(model_path: Path) -> None:
    import onnx
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model, full_check=True)
    logger.info("ONNX model validation passed")


def _simplify_onnx(model_path: Path) -> None:
    """Run onnx-simplifier (constant folding, dead code elimination)."""
    try:
        import onnx
        import onnxsim

        logger.info("Simplifying ONNX graph...")
        model = onnx.load(str(model_path))
        model_simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(model_simplified, str(model_path))
            logger.info("ONNX graph simplified")
        else:
            logger.warning("Simplification check failed, keeping original")
    except ImportError:
        logger.info("onnxsim not installed — skipping (pip install onnxsim)")


def _log_model_stats(model_path: Path) -> None:
    import onnx

    size_mb = model_path.stat().st_size / (1024 * 1024)
    model = onnx.load(str(model_path))
    num_nodes = len(model.graph.node)
    num_params = sum(np.prod(init.dims) for init in model.graph.initializer)
    logger.info(
        "Model stats: %.1f MB, %d nodes, %.1fM parameters",
        size_mb, num_nodes, num_params / 1e6,
    )
