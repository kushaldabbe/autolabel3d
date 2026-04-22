"""Factory functions for instantiating pipeline components from Hydra config.

The factory pattern decouples config from implementation — the orchestrator
calls build_detector(cfg.detector) and gets back a BaseDetector without
knowing which concrete class was instantiated.

To add a new implementation (e.g., a YOLO detector):
    1. Implement a class that extends BaseDetector
    2. Add an entry to DETECTOR_REGISTRY
    3. Create a config YAML at configs/detector/yolo.yaml
"""

from __future__ import annotations

import importlib
from typing import Any

from autolabel3d.data.base import BaseDataLoader
from autolabel3d.detection.base import BaseDetector
from autolabel3d.lifting.base import BaseLifter
from autolabel3d.segmentation.base import BaseSegmentor
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import registries — heavy model imports happen only when called
DATALOADER_REGISTRY: dict[str, tuple[str, str]] = {
    "nuscenes": ("autolabel3d.data.nuscenes_loader", "NuScenesLoader"),
    "dashcam":  ("autolabel3d.data.dashcam_loader",  "DashcamLoader"),
}

DETECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "grounding_dino": ("autolabel3d.detection.grounding_dino", "GroundingDINODetector"),
}

SEGMENTOR_REGISTRY: dict[str, tuple[str, str]] = {
    "sam2": ("autolabel3d.segmentation.sam2", "SAM2Segmentor"),
}

LIFTER_REGISTRY: dict[str, tuple[str, str]] = {
    "depth_anything": ("autolabel3d.lifting.depth_anything", "DepthAnythingLifter"),
    "boxer":          ("autolabel3d.lifting.boxer",          "BoxerLifter"),
}


def _import_class(module_path: str, class_name: str) -> type:
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _validate_name(name: str, registry: dict, kind: str) -> None:
    if name not in registry:
        raise ValueError(
            f"Unknown {kind}: '{name}'. Available: {sorted(registry.keys())}"
        )


def build_dataloader(cfg: Any) -> BaseDataLoader:
    name = cfg.name
    _validate_name(name, DATALOADER_REGISTRY, "data loader")
    module_path, class_name = DATALOADER_REGISTRY[name]
    instance = _import_class(module_path, class_name)(cfg)
    logger.info("Built data loader: %s", class_name)
    return instance


def build_detector(cfg: Any, device: Any = None) -> BaseDetector:
    name = cfg.name
    _validate_name(name, DETECTOR_REGISTRY, "detector")
    module_path, class_name = DETECTOR_REGISTRY[name]
    instance = _import_class(module_path, class_name)(cfg, device=device)
    logger.info("Built detector: %s", class_name)
    return instance


def build_segmentor(cfg: Any, device: Any = None) -> BaseSegmentor:
    name = cfg.name
    _validate_name(name, SEGMENTOR_REGISTRY, "segmentor")
    module_path, class_name = SEGMENTOR_REGISTRY[name]
    instance = _import_class(module_path, class_name)(cfg, device=device)
    logger.info("Built segmentor: %s", class_name)
    return instance


def build_lifter(cfg: Any) -> BaseLifter:
    name = cfg.name
    _validate_name(name, LIFTER_REGISTRY, "lifter")
    module_path, class_name = LIFTER_REGISTRY[name]
    instance = _import_class(module_path, class_name)(cfg)
    logger.info("Built lifter: %s", class_name)
    return instance
