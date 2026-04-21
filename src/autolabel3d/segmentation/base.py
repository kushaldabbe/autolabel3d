"""Abstract base class for video segmentation models.

Takes detection boxes as prompts and produces per-frame pixel masks
with temporal tracking (consistent track_id across frames via SAM 2 memory).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from autolabel3d.data.schemas import Frame, FrameDetections, FrameMasks


class BaseSegmentor(ABC):
    """Abstract video segmentation model."""

    @abstractmethod
    def segment_video(
        self,
        frames: list[Frame],
        initial_detections: FrameDetections,
    ) -> list[FrameMasks]:
        """Segment and track objects across an entire video clip.

        Args:
            frames: Ordered sequence of video frames.
            initial_detections: Detections on frames[0], used as box prompts.

        Returns:
            List of FrameMasks (one per frame) with consistent track_ids.
        """
        ...

    @abstractmethod
    def segment_frame(
        self,
        frame: Frame,
        detections: FrameDetections,
    ) -> FrameMasks:
        """Segment objects in a single frame (no temporal tracking)."""
        ...
