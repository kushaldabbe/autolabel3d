"""Hydra CLI entry point for the autolabel3d pipeline.

Usage:
    # Default config (nuScenes + Grounding DINO + SAM 2 + Depth Anything)
    python -m autolabel3d.cli

    # Override data source and lifter
    python -m autolabel3d.cli data=dashcam lifter=boxer

    # Override device and output dir
    python -m autolabel3d.cli pipeline.device=cpu pipeline.output_dir=outputs/run1

    # Limit frames (useful for testing)
    python -m autolabel3d.cli +max_frames=5

    # Disable video mode (per-frame processing)
    python -m autolabel3d.cli +video_mode=false
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from autolabel3d.factory import (
    build_dataloader,
    build_detector,
    build_lifter,
    build_segmentor,
)
from autolabel3d.pipeline import Pipeline
from autolabel3d.utils.device import get_device
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point — Hydra populates cfg from YAML + CLI overrides."""
    logger.info("=" * 60)
    logger.info("autolabel3d — AV Perception Auto-Labeling Pipeline")
    logger.info("=" * 60)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = get_device(cfg.pipeline.get("device", "auto"))
    logger.info("Device: %s", device)

    _set_seed(cfg.pipeline.get("seed", 42))

    logger.info("Building pipeline components...")
    t0 = time.perf_counter()
    data_loader = build_dataloader(cfg.data)
    detector    = build_detector(cfg.detector, device=device)
    segmentor   = build_segmentor(cfg.segmentor, device=device)
    lifter      = build_lifter(cfg.lifter)
    logger.info("Components built in %.2fs", time.perf_counter() - t0)

    output_dir = Path(cfg.pipeline.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = Path(hydra.utils.get_original_cwd()) / output_dir

    pipeline = Pipeline(
        data_loader=data_loader,
        detector=detector,
        segmentor=segmentor,
        lifter=lifter,
        output_dir=output_dir,
        video_mode=cfg.get("video_mode", True),
    )

    result = pipeline.run(
        max_frames=cfg.get("max_frames", None),
        iou_threshold=cfg.get("iou_threshold", 0.5),
    )

    print("\n" + result.summary())
    logger.info("Output saved to %s", output_dir)


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
