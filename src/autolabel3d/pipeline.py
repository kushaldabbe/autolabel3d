"""Pipeline orchestrator — wires all modules into an end-to-end auto-labeling flow.

Pipeline architecture:
    DataLoader → Detector → Segmentor → Lifter → [Evaluator]
      (frames)    (2D boxes)  (masks)    (3D boxes)  (mAP)

Two execution modes:

VIDEO MODE (default):
    Detect on the FIRST frame only → SAM 2 propagates masks through all frames
    via memory attention → lift masks to 3D per frame.
    Preferred: temporally consistent track IDs and smoother masks.

FRAME-BY-FRAME MODE (fallback):
    Detect → segment → lift independently per frame.
    Useful when temporal context is unavailable (random single-frame inference).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from autolabel3d.data.base import BaseDataLoader
from autolabel3d.data.schemas import BBox3D, Frame, FrameAnnotations, FrameMasks
from autolabel3d.detection.base import BaseDetector
from autolabel3d.evaluation.kitti_format import write_kitti_annotations
from autolabel3d.evaluation.metrics import EvaluationResult, evaluate
from autolabel3d.lifting.base import BaseLifter
from autolabel3d.segmentation.base import BaseSegmentor
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Complete result from a single pipeline run."""

    annotations: list[FrameAnnotations] = field(default_factory=list)
    evaluation: EvaluationResult | None = None
    timing: dict[str, float] = field(default_factory=dict)
    num_frames: int = 0
    num_objects: int = 0

    @property
    def fps(self) -> float:
        total = self.timing.get("total", 0)
        return self.num_frames / total if total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Pipeline: {self.num_frames} frames, {self.num_objects} objects",
            f"Throughput: {self.fps:.2f} FPS",
            "",
            "Timing breakdown:",
        ]
        total = max(self.timing.get("total", 1e-6), 1e-6)
        for stage, dur in self.timing.items():
            if stage != "total":
                lines.append(f"  {stage:20s}: {dur:7.2f}s ({100 * dur / total:5.1f}%)")
        lines.append(f"  {'total':20s}: {self.timing.get('total', 0):7.2f}s")
        if self.evaluation:
            lines += ["", self.evaluation.summary()]
        return "\n".join(lines)


class Pipeline:
    """End-to-end auto-labeling pipeline orchestrator."""

    def __init__(
        self,
        data_loader: BaseDataLoader,
        detector: BaseDetector,
        segmentor: BaseSegmentor,
        lifter: BaseLifter,
        output_dir: Path | None = None,
        video_mode: bool = True,
    ) -> None:
        self.data_loader = data_loader
        self.detector = detector
        self.segmentor = segmentor
        self.lifter = lifter
        self.output_dir = Path(output_dir) if output_dir else None
        self.video_mode = video_mode

        if self.output_dir:
            (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

        logger.info(
            "Pipeline ready: mode=%s, output=%s",
            "video" if video_mode else "frame", self.output_dir,
        )

    def run(
        self,
        ground_truth: list[list[BBox3D]] | None = None,
        iou_threshold: float = 0.5,
        max_frames: int | None = None,
    ) -> PipelineResult:
        """Execute the full auto-labeling pipeline.

        Args:
            ground_truth: Optional per-frame GT boxes for evaluation.
            iou_threshold: IoU threshold for evaluation.
            max_frames: Cap on frames to process (for debugging).

        Returns:
            PipelineResult with annotations, timing, and optional evaluation.
        """
        result = PipelineResult()
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        frames = self._load_frames(max_frames)
        result.timing["data_loading"] = time.perf_counter() - t0
        result.num_frames = len(frames)

        if not frames:
            logger.warning("No frames loaded.")
            result.timing["total"] = time.perf_counter() - total_start
            return result

        logger.info("Loaded %d frames", len(frames))

        if self.video_mode:
            annotations = self._run_video_mode(frames, result)
        else:
            annotations = self._run_frame_mode(frames, result)

        result.annotations = annotations
        result.num_objects = sum(a.num_annotations for a in annotations)

        if self.output_dir:
            t0 = time.perf_counter()
            self._save_annotations(annotations)
            result.timing["output"] = time.perf_counter() - t0

        if ground_truth is not None:
            t0 = time.perf_counter()
            preds = [a.boxes_3d for a in annotations]
            gt_aligned = ground_truth[: len(preds)]
            if len(gt_aligned) == len(preds):
                result.evaluation = evaluate(preds, gt_aligned, iou_threshold=iou_threshold)
                logger.info("mAP@%.1f = %.4f", iou_threshold, result.evaluation.map)
            else:
                logger.warning(
                    "GT frame count (%d) != pred count (%d), skipping eval",
                    len(gt_aligned), len(preds),
                )
            result.timing["evaluation"] = time.perf_counter() - t0

        result.timing["total"] = time.perf_counter() - total_start
        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_frames(self, max_frames: int | None) -> list[Frame]:
        frames: list[Frame] = []
        for frame in self.data_loader.load_frames():
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        return frames

    def _run_video_mode(
        self, frames: list[Frame], result: PipelineResult
    ) -> list[FrameAnnotations]:
        t0 = time.perf_counter()
        first_dets = self.detector.detect(frames[0])
        result.timing["detection"] = time.perf_counter() - t0
        logger.info("Detected %d objects in frame 0", first_dets.num_detections)

        if first_dets.num_detections == 0:
            logger.warning("No detections in first frame — empty annotations.")
            return [FrameAnnotations(frame_idx=f.frame_idx) for f in frames]

        t0 = time.perf_counter()
        all_masks = self.segmentor.segment_video(frames, first_dets)
        result.timing["segmentation"] = time.perf_counter() - t0
        logger.info(
            "Segmented %d frames (avg %.1f masks/frame)",
            len(all_masks),
            np.mean([m.num_masks for m in all_masks]) if all_masks else 0,
        )

        t0 = time.perf_counter()
        annotations = self._lift_all_frames(frames, all_masks)
        result.timing["lifting"] = time.perf_counter() - t0
        return annotations

    def _run_frame_mode(
        self, frames: list[Frame], result: PipelineResult
    ) -> list[FrameAnnotations]:
        annotations: list[FrameAnnotations] = []
        det_t = seg_t = lift_t = 0.0

        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            dets = self.detector.detect(frame)
            det_t += time.perf_counter() - t0

            t0 = time.perf_counter()
            masks = self.segmentor.segment_frame(frame, dets)
            seg_t += time.perf_counter() - t0

            t0 = time.perf_counter()
            boxes_3d = self._lift_frame(frame, masks)
            lift_t += time.perf_counter() - t0

            annotations.append(FrameAnnotations(
                frame_idx=frame.frame_idx,
                boxes_3d=boxes_3d,
                detections=dets,
                masks=masks,
            ))

            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                logger.info("Processed frame %d/%d", i + 1, len(frames))

        result.timing.update(detection=det_t, segmentation=seg_t, lifting=lift_t)
        return annotations

    def _lift_all_frames(
        self, frames: list[Frame], all_masks: list[FrameMasks]
    ) -> list[FrameAnnotations]:
        annotations = []
        for frame, frame_masks in zip(frames, all_masks, strict=True):
            boxes_3d = self._lift_frame(frame, frame_masks)
            annotations.append(FrameAnnotations(
                frame_idx=frame.frame_idx,
                boxes_3d=boxes_3d,
                masks=frame_masks,
            ))
        logger.info(
            "Lifted %d objects across %d frames",
            sum(a.num_annotations for a in annotations), len(annotations),
        )
        return annotations

    def _lift_frame(self, frame: Frame, frame_masks: FrameMasks) -> list[BBox3D]:
        if not frame_masks.masks:
            return []
        if frame.calibration is None:
            logger.warning("Frame %d has no calibration — skipping lift", frame.frame_idx)
            return []
        results = self.lifter.lift_batch(
            frame_masks.masks, frame.calibration, image=frame.image
        )
        return [b for b in results if b is not None]

    def _save_annotations(self, annotations: list[FrameAnnotations]) -> None:
        labels_dir = self.output_dir / "labels"
        for ann in annotations:
            write_kitti_annotations(ann.boxes_3d, labels_dir / f"{ann.frame_idx:06d}.txt")
        logger.info("Saved %d annotation files to %s", len(annotations), labels_dir)
