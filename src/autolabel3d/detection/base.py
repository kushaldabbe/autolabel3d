"""Abstract base class for 2D object detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from autolabel3d.data.schemas import Frame, FrameDetections


class BaseDetector(ABC):
    """Abstract 2D object detector."""

    @abstractmethod
    def detect(self, frame: Frame) -> FrameDetections:
        """Run detection on a single frame and return bounding boxes."""
        ...

    @abstractmethod
    def detect_batch(self, frames: list[Frame]) -> list[FrameDetections]:
        """Run detection on a batch of frames."""
        ...
