"""Grounding DINO open-vocabulary 2D object detector.

Architecture overview (the math behind the model):

Grounding DINO fuses a DETR-style vision transformer with a text encoder (BERT)
to perform open-vocabulary detection. The key innovation is cross-modality fusion
at multiple stages:

1. BACKBONE (Swin Transformer):
   Image I ∈ R^{H×W×3} → multi-scale features {F_l} where F_l ∈ R^{H_l × W_l × C_l}

2. TEXT ENCODER (BERT):
   Text prompt T = "car . pedestrian . cyclist . traffic cone ."
   → token embeddings E_text ∈ R^{L × D}
   The " . " separator tells the model these are separate class concepts.

3. FEATURE ENHANCER (Cross-Modality Fusion):
   Bidirectional cross-attention between image features and text tokens:
       Q_img = W_q · F_image,  K_txt = W_k · E_text,  V_txt = W_v · E_text
       Attn_img→txt = softmax(Q_img K_txt^T / √d_k) · V_txt

4. LANGUAGE-GUIDED QUERY SELECTION:
   Top-N image features by max text-alignment score are used as decoder queries:
       score_i = max_j(F_image_i · E_text_j^T)

5. CONTRASTIVE CLASSIFICATION (sigmoid, not softmax):
       score(query_i, class_j) = σ(query_i · text_class_j^T / τ)
   Open-vocabulary: no fixed class set; any text prompt works.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from autolabel3d.data.schemas import (
    Detection2D,
    Frame,
    FrameDetections,
    ObjectClass,
)
from autolabel3d.detection.base import BaseDetector
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

# Map text phrases returned by the model → our ObjectClass enum
PHRASE_TO_CLASS: dict[str, ObjectClass] = {
    "car":          ObjectClass.CAR,
    "pedestrian":   ObjectClass.PEDESTRIAN,
    "cyclist":      ObjectClass.CYCLIST,
    "traffic cone": ObjectClass.TRAFFIC_CONE,
    "traffic_cone": ObjectClass.TRAFFIC_CONE,
}


class GroundingDINODetector(BaseDetector):
    """Open-vocabulary 2D object detector using Grounding DINO.

    Uses the HuggingFace transformers implementation:
    - Works on CUDA, MPS, and CPU without custom compiled ops
    - Handles model downloading and caching automatically (~700 MB)

    Config keys (see configs/detector/grounding_dino.yaml):
        model_id:        HuggingFace model ID (IDEA-Research/grounding-dino-base)
        box_threshold:   Min objectness score to keep a box
        text_threshold:  Min text-alignment score for phrase matching
        text_prompt:     Override the auto-built prompt (optional)
        nms.enabled:     Apply per-class NMS post-detection
        nms.iou_threshold: IoU threshold for NMS suppression
    """

    def __init__(self, cfg: DictConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.box_threshold: float = cfg.box_threshold
        self.text_threshold: float = cfg.text_threshold
        self.nms_enabled: bool = cfg.nms.enabled
        self.nms_iou_threshold: float = cfg.nms.iou_threshold

        if device is None:
            from autolabel3d.utils.device import get_device
            device = get_device()
        self.device = device

        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load_model()
        return self._processor

    def _load_model(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = self.cfg.model_id
        logger.info("Loading Grounding DINO: %s", model_id)

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

        try:
            self._model = self._model.to(self.device)
        except RuntimeError as exc:
            logger.warning("Failed to move to %s: %s. Falling back to CPU.", self.device, exc)
            self.device = torch.device("cpu")
            self._model = self._model.to(self.device)

        self._model.eval()
        logger.info("Grounding DINO ready on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Frame, text_prompt: str | None = None) -> FrameDetections:
        """Detect objects in a single frame.

        Pipeline:
            1. BGR (OpenCV) → RGB → PIL Image
            2. Processor tokenises text + normalises image
            3. Forward pass through Grounding DINO
            4. Post-process: threshold → phrase mapping → optional NMS
        """
        image_rgb = frame.image[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        prompt = text_prompt or self._build_text_prompt()

        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # PIL size is (W, H); post_process expects (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        detections = self._parse_results(results, frame.frame_idx)

        if self.nms_enabled and detections.num_detections > 0:
            detections = self._apply_nms(detections)

        logger.debug(
            "Frame %d: %d detections (prompt: %r)",
            frame.frame_idx, detections.num_detections, prompt,
        )
        return detections

    def detect_batch(self, frames: list[Frame]) -> list[FrameDetections]:
        return [self.detect(frame) for frame in frames]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_text_prompt(self) -> str:
        """Build " . "-separated text prompt from config or defaults."""
        if self.cfg.get("text_prompt"):
            return self.cfg.text_prompt
        classes = list(PHRASE_TO_CLASS.keys())
        return " . ".join(classes) + " ."

    def _parse_results(self, results: dict, frame_idx: int) -> FrameDetections:
        """Convert HuggingFace post-processed results to our Detection2D schema."""
        boxes = results["boxes"].cpu().numpy()   # (N, 4) [x1,y1,x2,y2]
        scores = results["scores"].cpu().numpy() # (N,)
        # transformers ≥5.x uses "text_labels"; older uses "labels"
        labels: list[str] = results.get("text_labels", results.get("labels", []))

        detections: list[Detection2D] = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            phrase = label.strip().lower()
            obj_class = self._map_phrase_to_class(phrase)
            if obj_class is None:
                continue

            detections.append(Detection2D(
                bbox=box.astype(np.float32),
                confidence=float(score),
                class_name=obj_class,
                class_phrase=phrase,
            ))

        return FrameDetections(frame_idx=frame_idx, detections=detections)

    @staticmethod
    def _map_phrase_to_class(phrase: str) -> ObjectClass | None:
        """Map a model phrase to an ObjectClass (exact, then substring match)."""
        if phrase in PHRASE_TO_CLASS:
            return PHRASE_TO_CLASS[phrase]
        for key, cls in PHRASE_TO_CLASS.items():
            if key in phrase:
                return cls
        return None

    def _apply_nms(self, detections: FrameDetections) -> FrameDetections:
        """Per-class Non-Maximum Suppression.

        NMS algorithm (per class):
            1. Sort by confidence descending.
            2. Greedily keep the top box.
            3. Remove all remaining boxes with IoU > threshold.
            4. Repeat until empty.

        IoU(A, B) = |A ∩ B| / (|A| + |B| - |A ∩ B|)

        Per-class NMS prevents a car detection from suppressing a nearby pedestrian.
        """
        boxes = detections.boxes  # (N, 4)
        scores = np.array([d.confidence for d in detections.detections])
        class_ids = np.array([d.class_name.value for d in detections.detections])

        keep_indices: list[int] = []

        for cls_val in np.unique(class_ids):
            cls_mask = class_ids == cls_val
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            order = cls_scores.argsort()[::-1]
            while len(order) > 0:
                best = order[0]
                keep_indices.append(int(cls_indices[best]))
                if len(order) == 1:
                    break
                ious = self._compute_iou_1_vs_n(cls_boxes[best], cls_boxes[order[1:]])
                remaining = np.where(ious < self.nms_iou_threshold)[0]
                order = order[remaining + 1]

        kept = [detections.detections[i] for i in sorted(keep_indices)]
        return FrameDetections(frame_idx=detections.frame_idx, detections=kept)

    @staticmethod
    def _compute_iou_1_vs_n(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """IoU of one box (4,) against N boxes (N, 4), all in [x1,y1,x2,y2]."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter
        return np.where(union > 0.0, inter / union, 0.0)
