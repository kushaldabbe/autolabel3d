"""SAM 2.1 video segmentation with temporal tracking.

Architecture overview:

SAM 2 extends SAM from single images to video via a memory mechanism —
past frame features and object pointers are stored in a memory bank so
the model can propagate segmentation masks through time.

1. IMAGE ENCODER (Hiera — Hierarchical Vision Transformer):
   Multi-scale ViT with mask-unit attention (local at high-res, global at low-res).
   Frame I_t → multi-scale features {F_t^l}

2. MEMORY ATTENTION (core innovation):
   Current frame features attend to memory bank of past frames:
       F'_t = CrossAttention(Q=F_t, KV=[M_{past}; P_{objects}])
   Fixed-capacity bank (default 7 frames); oldest evicted first, except
   prompted frames which are always retained.

3. MASK DECODER:
   Transformer decoder that cross-attends from prompt tokens to image features.
   Outputs binary mask logits, IoU confidence, and an object pointer.

4. PROPAGATION LOOP (propagate_in_video):
   For each frame: encode → memory-attend → decode → store memory.
   Yields (frame_idx, obj_ids, mask_logits) — masks are raw logits.

Usage (primary mode):
    segmentor = SAM2Segmentor(cfg)
    frame_masks = segmentor.segment_video(frames, initial_detections)

Box prompts from Grounding DINO are fed on frame 0; SAM 2 propagates
across all subsequent frames with stable track IDs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from autolabel3d.data.schemas import (
    Frame,
    FrameDetections,
    FrameMasks,
    ObjectClass,
    SegmentationMask,
)
from autolabel3d.segmentation.base import BaseSegmentor
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

SAM2_MODEL_IDS: dict[str, str] = {
    "tiny":      "facebook/sam2.1-hiera-tiny",
    "small":     "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large":     "facebook/sam2.1-hiera-large",
}


class SAM2Segmentor(BaseSegmentor):
    """SAM 2.1 video segmentor with temporal object tracking.

    Two modes:

    VIDEO MODE (segment_video):
        - Writes frames to a temp dir (SAM 2's required input format)
        - Box prompts on frame 0 → propagate_in_video across all frames
        - Returns temporally consistent masks with stable track IDs
        - Primary mode for the pipeline

    FRAME MODE (segment_frame):
        - Uses SAM2ImagePredictor (no temporal context)
        - Useful for testing and single-frame annotation
    """

    def __init__(self, cfg: DictConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.model_size: str = cfg.model_size
        self.min_mask_area: int = cfg.propagation.min_mask_area

        if device is None:
            from autolabel3d.utils.device import get_device
            device = get_device()
        self.device = device

        self._video_predictor = None
        self._image_predictor = None

    @property
    def model_id(self) -> str:
        if self.model_size not in SAM2_MODEL_IDS:
            raise ValueError(
                f"Unknown SAM 2 model size: {self.model_size!r}. "
                f"Choose from: {list(SAM2_MODEL_IDS.keys())}"
            )
        return SAM2_MODEL_IDS[self.model_size]

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def video_predictor(self):
        if self._video_predictor is None:
            self._load_video_predictor()
        return self._video_predictor

    @property
    def image_predictor(self):
        if self._image_predictor is None:
            self._load_image_predictor()
        return self._image_predictor

    def _load_video_predictor(self) -> None:
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        logger.info("Loading SAM 2 video predictor: %s", self.model_id)
        try:
            self._video_predictor = SAM2VideoPredictor.from_pretrained(
                self.model_id, device=str(self.device),
            )
        except (RuntimeError, AssertionError) as exc:
            logger.warning("Failed on %s: %s — falling back to CPU.", self.device, exc)
            self.device = torch.device("cpu")
            self._video_predictor = SAM2VideoPredictor.from_pretrained(
                self.model_id, device="cpu",
            )
        logger.info("SAM 2 video predictor ready on %s", self.device)

    def _load_image_predictor(self) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info("Loading SAM 2 image predictor: %s", self.model_id)
        try:
            self._image_predictor = SAM2ImagePredictor.from_pretrained(
                self.model_id, device=str(self.device),
            )
        except (RuntimeError, AssertionError) as exc:
            logger.warning("Failed on %s: %s — falling back to CPU.", self.device, exc)
            self.device = torch.device("cpu")
            self._image_predictor = SAM2ImagePredictor.from_pretrained(
                self.model_id, device="cpu",
            )
        logger.info("SAM 2 image predictor ready on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_video(
        self,
        frames: list[Frame],
        initial_detections: FrameDetections,
    ) -> list[FrameMasks]:
        """Segment and track objects across a video clip.

        Algorithm:
            1. Write frames to a temp JPEG directory (SAM 2 reads from disk)
            2. init_state: builds internal frame index
            3. add_new_points_or_box: register each detection box as a tracked object
            4. propagate_in_video: memory-attention loop → (frame_idx, obj_ids, logits)
            5. Threshold logits → binary masks; filter by min_mask_area

        Args:
            frames: Ordered list of video frames.
            initial_detections: Detections on frames[0] used as box prompts.

        Returns:
            List of FrameMasks (one per frame) with consistent track_ids.
        """
        if not frames:
            return []

        if initial_detections.num_detections == 0:
            logger.warning("No detections to prompt SAM 2 — returning empty masks.")
            return [FrameMasks(frame_idx=f.frame_idx) for f in frames]

        predictor = self.video_predictor

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir)
            # SAM 2 expects numerically sorted filenames
            for i, frame in enumerate(frames):
                cv2.imwrite(str(frame_dir / f"{i:06d}.jpg"), frame.image)

            inference_state = predictor.init_state(video_path=str(frame_dir))

            obj_id_to_class: dict[int, ObjectClass] = {}
            obj_id_to_confidence: dict[int, float] = {}

            for obj_id, det in enumerate(initial_detections.detections):
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=det.bbox,
                )
                obj_id_to_class[obj_id] = det.class_name
                obj_id_to_confidence[obj_id] = det.confidence

            logger.info(
                "Tracking %d objects across %d frames",
                len(obj_id_to_class), len(frames),
            )

            frame_results: dict[int, list[SegmentationMask]] = {
                i: [] for i in range(len(frames))
            }

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                # mask_logits: (N_objects, 1, H, W) — raw logits
                masks_prob = torch.sigmoid(mask_logits)

                for i, obj_id in enumerate(obj_ids):
                    obj_id = int(obj_id)
                    if obj_id not in obj_id_to_class:
                        continue

                    # logit > 0  ↔  sigmoid > 0.5
                    mask_binary = (masks_prob[i, 0] > 0.5).cpu().numpy()
                    if mask_binary.sum() < self.min_mask_area:
                        continue

                    frame_results[frame_idx].append(SegmentationMask(
                        mask=mask_binary,
                        track_id=obj_id,
                        confidence=obj_id_to_confidence.get(obj_id, 1.0),
                        class_name=obj_id_to_class[obj_id],
                        bbox=self._mask_to_bbox(mask_binary),
                    ))

        return [
            FrameMasks(
                frame_idx=frames[i].frame_idx,
                masks=frame_results.get(i, []),
            )
            for i in range(len(frames))
        ]

    def segment_frame(
        self,
        frame: Frame,
        detections: FrameDetections,
    ) -> FrameMasks:
        """Segment objects in a single frame without temporal context.

        Uses SAM2ImagePredictor; box prompts from detections.
        """
        if detections.num_detections == 0:
            return FrameMasks(frame_idx=frame.frame_idx)

        predictor = self.image_predictor
        predictor.set_image(frame.image[:, :, ::-1])  # BGR → RGB

        masks_out: list[SegmentationMask] = []

        for obj_id, det in enumerate(detections.detections):
            masks, scores, _ = predictor.predict(
                box=det.bbox,
                multimask_output=False,
            )
            mask_binary = masks[0].astype(np.bool_)

            if mask_binary.sum() < self.min_mask_area:
                continue

            masks_out.append(SegmentationMask(
                mask=mask_binary,
                track_id=obj_id,
                confidence=float(scores[0]),
                class_name=det.class_name,
                bbox=self._mask_to_bbox(mask_binary),
            ))

        return FrameMasks(frame_idx=frame.frame_idx, masks=masks_out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> np.ndarray:
        """Compute tight [x1, y1, x2, y2] box from a binary mask (H, W)."""
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array(
            [cols.min(), rows.min(), cols.max(), rows.max()],
            dtype=np.float32,
        )
