"""Per-class 3D detection metrics: Precision, Recall, AP, mAP.

mAP OVERVIEW:
    TP: prediction matched to a GT box with IoU ≥ threshold
    FP: prediction with no matching GT
    FN: GT box with no matching prediction

Matching (PASCAL VOC greedy protocol):
    Sort predictions by confidence (descending). For each prediction find the
    highest-IoU unmatched GT box. If IoU ≥ threshold AND box unmatched → TP,
    else → FP.

AP: Area under the Precision-Recall curve (all-point / COCO-style interpolation)
mAP: Mean AP across all evaluated classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import BBox3D, ObjectClass
from autolabel3d.evaluation.iou import compute_iou_3d, compute_iou_bev
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassMetrics:
    """Per-class evaluation results."""

    class_name: ObjectClass
    iou_threshold: float
    num_predictions: int = 0
    num_ground_truth: int = 0
    num_true_positive: int = 0
    num_false_positive: int = 0
    num_false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    ap: float = 0.0
    precisions: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0


@dataclass
class EvaluationResult:
    """Aggregated evaluation across all classes."""

    iou_threshold: float
    use_3d_iou: bool
    per_class: dict[ObjectClass, ClassMetrics] = field(default_factory=dict)

    @property
    def map(self) -> float:
        return float(np.mean([m.ap for m in self.per_class.values()])) if self.per_class else 0.0

    @property
    def mean_precision(self) -> float:
        return float(np.mean([m.precision for m in self.per_class.values()])) if self.per_class else 0.0

    @property
    def mean_recall(self) -> float:
        return float(np.mean([m.recall for m in self.per_class.values()])) if self.per_class else 0.0

    def summary(self) -> str:
        mode = "3D" if self.use_3d_iou else "BEV"
        lines = [
            f"=== Evaluation (IoU={self.iou_threshold:.1f}, {mode}) ===",
            f"mAP={self.map:.4f}  Mean-P={self.mean_precision:.4f}  Mean-R={self.mean_recall:.4f}",
            "",
        ]
        for cls, m in sorted(self.per_class.items(), key=lambda kv: kv[0].value):
            lines.append(
                f"  {cls.value:15s}  AP={m.ap:.4f}  P={m.precision:.4f}  "
                f"R={m.recall:.4f}  F1={m.f1:.4f}  "
                f"(TP={m.num_true_positive} FP={m.num_false_positive} FN={m.num_false_negative})"
            )
        return "\n".join(lines)


def evaluate(
    predictions: list[list[BBox3D]],
    ground_truths: list[list[BBox3D]],
    iou_threshold: float = 0.5,
    use_3d_iou: bool = True,
    classes: list[ObjectClass] | None = None,
) -> EvaluationResult:
    """Evaluate 3D detections against ground truth.

    Args:
        predictions: Per-frame predicted boxes.
        ground_truths: Per-frame GT boxes (same length).
        iou_threshold: IoU threshold for TP matching.
        use_3d_iou: True = full 3D IoU; False = BEV IoU.
        classes: Classes to evaluate. None = all classes in GT.

    Returns:
        EvaluationResult with per-class AP, mAP, P, R.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Frame count mismatch: {len(predictions)} preds vs {len(ground_truths)} GT"
        )

    iou_fn = compute_iou_3d if use_3d_iou else compute_iou_bev

    if classes is None:
        all_cls: set[ObjectClass] = set()
        for frame_gt in ground_truths:
            for b in frame_gt:
                all_cls.add(b.class_name)
        classes = sorted(all_cls, key=lambda c: c.value)

    result = EvaluationResult(iou_threshold=iou_threshold, use_3d_iou=use_3d_iou)
    for cls in classes:
        result.per_class[cls] = _evaluate_class(
            predictions, ground_truths, cls, iou_threshold, iou_fn
        )

    logger.info("Eval mAP@%.1f=%s = %.4f", iou_threshold, "3D" if use_3d_iou else "BEV", result.map)
    return result


def _evaluate_class(
    predictions: list[list[BBox3D]],
    ground_truths: list[list[BBox3D]],
    target_class: ObjectClass,
    iou_threshold: float,
    iou_fn,
) -> ClassMetrics:
    """PASCAL VOC AP for one class."""
    all_preds: list[tuple[int, BBox3D]] = [
        (fi, b)
        for fi, fp in enumerate(predictions)
        for b in fp
        if b.class_name == target_class
    ]
    gt_per_frame: dict[int, list[BBox3D]] = {
        fi: [b for b in fg if b.class_name == target_class]
        for fi, fg in enumerate(ground_truths)
        if any(b.class_name == target_class for b in fg)
    }
    total_gt = sum(len(v) for v in gt_per_frame.values())

    all_preds.sort(key=lambda x: x[1].confidence, reverse=True)
    gt_matched = {fi: [False] * len(gts) for fi, gts in gt_per_frame.items()}

    tp_list, fp_list = [], []
    for frame_idx, pred_box in all_preds:
        if frame_idx not in gt_per_frame:
            tp_list.append(0); fp_list.append(1)
            continue

        gt_boxes = gt_per_frame[frame_idx]
        best_iou, best_gi = 0.0, -1
        for gi, gt_box in enumerate(gt_boxes):
            iou = iou_fn(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_iou >= iou_threshold and not gt_matched[frame_idx][best_gi]:
            tp_list.append(1); fp_list.append(0)
            gt_matched[frame_idx][best_gi] = True
        else:
            tp_list.append(0); fp_list.append(1)

    tp_cs = np.cumsum(tp_list, dtype=np.float64)
    fp_cs = np.cumsum(fp_list, dtype=np.float64)
    recalls = tp_cs / max(total_gt, 1)
    precisions = tp_cs / (tp_cs + fp_cs)

    ap = _compute_ap(precisions, recalls)
    total_tp = int(tp_cs[-1]) if len(tp_cs) else 0
    total_fp = int(fp_cs[-1]) if len(fp_cs) else 0

    return ClassMetrics(
        class_name=target_class,
        iou_threshold=iou_threshold,
        num_predictions=len(all_preds),
        num_ground_truth=total_gt,
        num_true_positive=total_tp,
        num_false_positive=total_fp,
        num_false_negative=total_gt - total_tp,
        precision=float(precisions[-1]) if len(precisions) else 0.0,
        recall=float(recalls[-1]) if len(recalls) else 0.0,
        ap=ap,
        precisions=precisions.tolist(),
        recalls=recalls.tolist(),
    )


def _compute_ap(
    precisions: NDArray[np.float64],
    recalls: NDArray[np.float64],
) -> float:
    """All-point interpolated AP (COCO-style).

    Monotonically decreases precision from right to left, then integrates:
        AP = Σᵢ (r_{i+1} − rᵢ) × p_interp(r_{i+1})
    """
    if len(precisions) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
