#!/usr/bin/env python3
"""Demo script — run the autolabel3d pipeline on any image or video.

Usage:
    # Single image
    python scripts/demo.py --input data/sample.jpg

    # Video (every 5th frame, max 30 frames)
    python scripts/demo.py --input data/dashcam.mp4 --every-n 5 --max-frames 30

    # Use Boxer lifter (no depth model needed)
    python scripts/demo.py --input data/sample.jpg --lifter boxer

    # Full options
    python scripts/demo.py \\
        --input data/dashcam.mp4 \\
        --output outputs/demo \\
        --max-frames 20 \\
        --every-n 3 \\
        --device cuda \\
        --lifter depth_anything \\
        --detector-threshold 0.3 \\
        --sam-model-size tiny
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autolabel3d.data.schemas import (
    BBox3D,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    Frame,
    FrameAnnotations,
    FrameDetections,
    FrameMasks,
)
from autolabel3d.evaluation.kitti_format import write_kitti_annotations
from autolabel3d.utils.logging import get_logger
from autolabel3d.visualization.bev import draw_bev
from autolabel3d.visualization.overlay import CLASS_COLORS, DEFAULT_COLOR, draw_detections, draw_masks

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="autolabel3d demo — full pipeline on any image or video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Path to image or video file")
    p.add_argument("--output", "-o", default="outputs/demo", help="Output directory")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--every-n", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--lifter", default="depth_anything", choices=["depth_anything", "boxer"])
    p.add_argument("--detector-threshold", type=float, default=0.3)
    p.add_argument("--sam-model-size", default="tiny",
                   choices=["tiny", "small", "base_plus", "large"])
    p.add_argument("--depth-model-size", default="small", choices=["small", "base", "large"])
    p.add_argument("--video-fps", type=float, default=6.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _make_calibration(h: int, w: int) -> CameraCalibration:
    """Default dashcam calibration assuming ~90° horizontal FOV."""
    fx = fy = float(w) / 2.0
    return CameraCalibration(
        intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0),
        extrinsics=CameraExtrinsics(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        ),
    )


def load_frames_from_image(path: Path) -> list[Frame]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    return [Frame(image=image, frame_idx=0, timestamp=0.0,
                  camera_name="demo", calibration=_make_calibration(h, w), source_path=path)]


def load_frames_from_video(path: Path, every_n: int, max_frames: int) -> list[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = list(range(0, total, every_n))[:max_frames]
    logger.info("Video: %s | %d frames @ %.1f FPS | sampling %d (every %d)",
                path.name, total, fps, len(indices), every_n)

    frames: list[Frame] = []
    calibration = None

    for out_idx, vid_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, image = cap.read()
        if not ret:
            continue
        if calibration is None:
            h, w = image.shape[:2]
            calibration = _make_calibration(h, w)
        frames.append(Frame(image=image, frame_idx=out_idx,
                            timestamp=vid_idx / fps, camera_name="demo",
                            calibration=calibration, source_path=path))

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_detector(args: argparse.Namespace, device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "name": "grounding_dino",
        "model_id": "IDEA-Research/grounding-dino-base",
        "weights_path": None,
        "box_threshold": args.detector_threshold,
        "text_threshold": 0.25,
        "text_prompt": None,
        "nms": {"enabled": True, "iou_threshold": 0.5},
    })
    from autolabel3d.detection.grounding_dino import GroundingDINODetector
    return GroundingDINODetector(cfg, device=device)


def build_segmentor(args: argparse.Namespace, device):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "name": "sam2",
        "model_size": args.sam_model_size,
        "checkpoint_dir": "weights/sam2",
        "propagation": {
            "max_frames_per_clip": 100,
            "overlap_frames": 10,
            "min_mask_area": 100,
        },
        "prompt_type": "box",
    })
    from autolabel3d.segmentation.sam2 import SAM2Segmentor
    return SAM2Segmentor(cfg, device=device)


def build_lifter(args: argparse.Namespace):
    from omegaconf import OmegaConf
    if args.lifter == "depth_anything":
        cfg = OmegaConf.create({
            "name": "depth_anything",
            "model_size": args.depth_model_size,
            "weights_path": "weights/depth_anything_v2",
            "fitting": {
                "method": "pca",
                "min_points": 50,
                "height_prior": {
                    "car": 1.5, "pedestrian": 1.7, "cyclist": 1.7, "traffic_cone": 0.8,
                },
            },
        })
        from autolabel3d.lifting.depth_anything import DepthAnythingLifter
        return DepthAnythingLifter(cfg)
    else:
        cfg = OmegaConf.create({
            "name": "boxer",
            "class_dimensions": {
                "car":          {"width": 1.8, "height": 1.5, "length": 4.5},
                "pedestrian":   {"width": 0.6, "height": 1.7, "length": 0.6},
                "cyclist":      {"width": 0.6, "height": 1.7, "length": 1.8},
                "traffic_cone": {"width": 0.3, "height": 0.8, "length": 0.3},
            },
        })
        from autolabel3d.lifting.boxer import BoxerLifter
        return BoxerLifter(cfg)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _draw_projected_3d_boxes(
    image: np.ndarray,
    boxes: list[BBox3D],
    calibration: CameraCalibration,
) -> np.ndarray:
    """Project 3D box corners onto image and draw wireframes."""
    vis = image.copy()
    K = calibration.intrinsics.matrix
    h, w = image.shape[:2]
    thickness = max(2, int(w / 640))
    thin = max(1, thickness // 2)
    font_scale = max(0.5, w / 2400)

    for box in boxes:
        color = CLASS_COLORS.get(box.class_name, DEFAULT_COLOR)
        front_color = tuple(min(255, int(c * 1.4)) for c in color)
        corners_3d = box.corners()  # (8, 3)

        projected = (K @ corners_3d.T).T
        z_vals = projected[:, 2]
        if np.any(z_vals <= 0):
            continue

        pts = (projected[:, :2] / z_vals[:, np.newaxis]).astype(int)
        if np.all(pts[:, 0] < -w) or np.all(pts[:, 0] > 2 * w):
            continue

        # Front face (brighter, thicker)
        for i, j in [(0, 1), (4, 5), (0, 4), (1, 5)]:
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), front_color, thickness, cv2.LINE_AA)
        # Back face
        for i, j in [(2, 3), (6, 7), (2, 6), (3, 7)]:
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, thin, cv2.LINE_AA)
        # Side edges
        for i, j in [(1, 2), (5, 6), (0, 3), (4, 7)]:
            cv2.line(vis, tuple(pts[i]), tuple(pts[j]), color, thin, cv2.LINE_AA)

        top_center = pts[4:8].mean(axis=0).astype(int)
        dist = float(np.linalg.norm(box.center))
        label = f"{box.class_name.value} {dist:.0f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thin)
        org = (int(top_center[0]) - tw // 2, int(top_center[1]) - 5)
        cv2.rectangle(vis, (org[0] - 2, org[1] - th - 4), (org[0] + tw + 2, org[1] + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, front_color, thin, cv2.LINE_AA)

    return vis


def save_visualizations(
    output_dir: Path,
    frame: Frame,
    detections: FrameDetections | None,
    masks: FrameMasks | None,
    boxes_3d: list[BBox3D],
    frame_idx: int,
) -> None:
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{frame_idx:06d}"
    image = frame.image

    if detections and detections.num_detections > 0:
        cv2.imwrite(str(viz_dir / f"{prefix}_detections.jpg"), draw_detections(image, detections))

    if masks and masks.num_masks > 0:
        cv2.imwrite(str(viz_dir / f"{prefix}_masks.jpg"), draw_masks(image, masks))

    if detections and masks:
        combined = draw_masks(image, masks)
        combined = draw_detections(combined, detections)
        cv2.imwrite(str(viz_dir / f"{prefix}_combined.jpg"), combined)

    if boxes_3d:
        cv2.imwrite(str(viz_dir / f"{prefix}_bev.jpg"), draw_bev(boxes_3d))

    if boxes_3d and frame.calibration:
        proj = _draw_projected_3d_boxes(image, boxes_3d, frame.calibration)
        cv2.imwrite(str(viz_dir / f"{prefix}_3d_projected.jpg"), proj)


def render_frame_panel(
    frame: Frame,
    detections: FrameDetections | None,
    masks: FrameMasks | None,
    boxes_3d: list[BBox3D],
    frame_num: int,
    total_frames: int,
) -> np.ndarray:
    """Render a 2×2 panel: original | masks+dets | 3D projected | BEV."""
    image = frame.image
    panel_w, panel_h = 960, 540

    def resize(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    # Panel 1 — original
    p1 = image.copy()
    cv2.putText(p1, f"Frame {frame_num + 1}/{total_frames}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(p1, "ORIGINAL", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    p1 = resize(p1)

    # Panel 2 — masks + detections
    if masks and masks.num_masks > 0:
        p2 = draw_masks(image, masks)
        if detections and detections.num_detections > 0:
            p2 = draw_detections(p2, detections)
    elif detections and detections.num_detections > 0:
        p2 = draw_detections(image, detections)
    else:
        p2 = image.copy()
    cv2.putText(p2, "MASKS + DETECTIONS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    p2 = resize(p2)

    # Panel 3 — 3D projected
    p3 = (_draw_projected_3d_boxes(image, boxes_3d, frame.calibration)
          if boxes_3d and frame.calibration else image.copy())
    cv2.putText(p3, f"3D BOXES (projected) — {len(boxes_3d)} objects",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    p3 = resize(p3)

    # Panel 4 — BEV
    p4 = (cv2.resize(draw_bev(boxes_3d), (panel_w, panel_h))
          if boxes_3d else np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
    cv2.putText(p4, "BEV (top-down)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

def _encode_ffmpeg(src: Path, dst: Path, fps: float, w: int, h: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(src / "%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
         "-vf", f"scale={w}:{h}", str(dst)],
        capture_output=True, timeout=120,
    )
    return result.returncode == 0


def _encode_opencv(src: Path, dst: Path, fps: float, w: int, h: int) -> bool:
    frames = sorted(src.glob("*.png"))
    if not frames:
        return False
    for codec in ["avc1", "mp4v"]:
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if not writer.isOpened():
            continue
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                writer.write(cv2.resize(img, (w, h)))
        writer.release()
        if dst.exists() and dst.stat().st_size > 1000:
            return True
    return False


def write_output_videos(
    output_dir: Path,
    frames: list[Frame],
    annotations: list[FrameAnnotations],
    first_det: FrameDetections,
    fps: float = 6.0,
) -> list[Path]:
    vid_dir = output_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].image.shape[:2]
    out_w = (min(w, 1920) // 2) * 2
    out_h = (int(h * out_w / w) // 2) * 2
    panel_w, panel_h = 1920, 1080

    total = len(frames)
    print(f"  Rendering {total} frames into output videos...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        panel_dir = tmp / "panel"; panel_dir.mkdir()
        masks_dir = tmp / "masks"; masks_dir.mkdir()
        boxes_dir = tmp / "boxes"; boxes_dir.mkdir()

        for i, (frame, ann) in enumerate(zip(frames, annotations)):
            det = ann.detections or (first_det if i == 0 else None)
            masks = ann.masks
            boxes = ann.boxes_3d

            cv2.imwrite(str(panel_dir / f"{i:06d}.png"),
                        render_frame_panel(frame, det, masks, boxes, i, total))

            mask_frame = draw_masks(frame.image, masks) if masks and masks.num_masks > 0 else frame.image.copy()
            if det and det.num_detections > 0:
                mask_frame = draw_detections(mask_frame, det)
            cv2.imwrite(str(masks_dir / f"{i:06d}.png"),
                        cv2.resize(mask_frame, (out_w, out_h)))

            box_frame = (_draw_projected_3d_boxes(frame.image, boxes, frame.calibration)
                         if boxes and frame.calibration else frame.image.copy())
            cv2.putText(box_frame, f"{len(boxes)} objects | Frame {i + 1}/{total}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imwrite(str(boxes_dir / f"{i:06d}.png"),
                        cv2.resize(box_frame, (out_w, out_h)))

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"    Rendered {i + 1}/{total} frames")

        use_ffmpeg = shutil.which("ffmpeg") is not None
        print(f"  Encoding with {'ffmpeg (H.264)' if use_ffmpeg else 'OpenCV'}...")
        created: list[Path] = []
        for src_dir, name, vw, vh in [
            (panel_dir, "combined_panel.mp4", panel_w, panel_h),
            (masks_dir, "masks_overlay.mp4",  out_w,   out_h),
            (boxes_dir, "boxes_3d.mp4",        out_w,   out_h),
        ]:
            dst = vid_dir / name
            ok = (_encode_ffmpeg(src_dir, dst, fps, vw, vh)
                  if use_ffmpeg else _encode_opencv(src_dir, dst, fps, vw, vh))
            if ok and dst.exists():
                created.append(dst)
            else:
                print(f"    WARNING: Failed to encode {name}")

    return created


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        sys.exit(1)

    from autolabel3d.utils.device import get_device
    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"\n{'='*60}\nLoading input: {input_path.name}\n{'='*60}")
    if _is_image(input_path):
        frames = load_frames_from_image(input_path)
        use_video_mode = False
    elif _is_video(input_path):
        frames = load_frames_from_video(input_path, args.every_n, args.max_frames)
        use_video_mode = len(frames) > 1
    else:
        print(f"Error: Unsupported file type: {input_path.suffix}")
        sys.exit(1)

    if not frames:
        print("Error: No frames loaded")
        sys.exit(1)
    print(f"Loaded {len(frames)} frame(s) — {frames[0].width}×{frames[0].height}")

    print(f"\n{'='*60}\nLoading models...\n{'='*60}")
    t0 = time.perf_counter()
    detector  = build_detector(args, device)
    segmentor = build_segmentor(args, device)
    lifter    = build_lifter(args)
    print(f"Models ready in {time.perf_counter() - t0:.1f}s")

    # --- Detection ---
    print(f"\n{'='*60}\nStage 1: Detection\n{'='*60}")
    t0 = time.perf_counter()
    if use_video_mode:
        first_det = detector.detect(frames[0])
        all_detections = [first_det] + [None] * (len(frames) - 1)
    else:
        all_detections = [detector.detect(f) for f in frames]
        first_det = all_detections[0]
    det_time = time.perf_counter() - t0
    print(f"  {first_det.num_detections} objects in frame 0 | {det_time:.2f}s")
    for d in first_det.detections:
        print(f"    - {d.class_name.value}: {d.confidence:.2f}")

    if first_det.num_detections == 0:
        print("\nNo objects detected. Try lowering --detector-threshold.")
        return

    # --- Segmentation ---
    print(f"\n{'='*60}\nStage 2: Segmentation\n{'='*60}")
    t0 = time.perf_counter()
    if use_video_mode:
        all_masks = segmentor.segment_video(frames, first_det)
    else:
        all_masks = [segmentor.segment_frame(f, d) for f, d in zip(frames, all_detections) if d]
    seg_time = time.perf_counter() - t0
    print(f"  {sum(m.num_masks for m in all_masks)} masks | {seg_time:.2f}s")

    # --- 3D Lifting ---
    print(f"\n{'='*60}\nStage 3: 3D Lifting ({args.lifter})\n{'='*60}")
    t0 = time.perf_counter()
    all_annotations: list[FrameAnnotations] = []
    for i, (frame, frame_masks) in enumerate(zip(frames, all_masks)):
        if not frame_masks.masks or frame.calibration is None:
            all_annotations.append(FrameAnnotations(frame_idx=frame.frame_idx))
            continue
        boxes_3d = [b for b in lifter.lift_batch(frame_masks.masks, frame.calibration,
                                                  image=frame.image) if b is not None]
        all_annotations.append(FrameAnnotations(
            frame_idx=frame.frame_idx, boxes_3d=boxes_3d,
            detections=all_detections[i], masks=frame_masks,
        ))
    lift_time = time.perf_counter() - t0
    total_boxes = sum(len(a.boxes_3d) for a in all_annotations)
    print(f"  {total_boxes} 3D boxes | {lift_time:.2f}s")

    # --- Save ---
    print(f"\n{'='*60}\nSaving outputs to {output_dir}\n{'='*60}")
    for frame, ann in zip(frames, all_annotations):
        if ann.boxes_3d:
            write_kitti_annotations(ann.boxes_3d, output_dir / "labels" / f"{frame.frame_idx:06d}.txt")
        save_visualizations(output_dir, frame, ann.detections, ann.masks, ann.boxes_3d, frame.frame_idx)
        cv2.imwrite(str(output_dir / "visualizations" / f"{frame.frame_idx:06d}_original.jpg"), frame.image)

    # --- Videos ---
    if len(frames) > 1:
        print(f"\n{'='*60}\nRendering output videos\n{'='*60}")
        video_files = write_output_videos(output_dir, frames, all_annotations, first_det, fps=args.video_fps)
        for vf in video_files:
            print(f"    {vf.name} ({vf.stat().st_size / 1e6:.1f} MB)")
    else:
        video_files = []

    # --- Summary ---
    total_time = det_time + seg_time + lift_time
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  Frames:       {len(frames)}")
    print(f"  Objects:      {total_boxes}")
    print(f"  Detection:    {det_time:.2f}s")
    print(f"  Segmentation: {seg_time:.2f}s")
    print(f"  Lifting:      {lift_time:.2f}s")
    print(f"  Total:        {total_time:.2f}s  ({len(frames)/total_time:.2f} FPS)")
    print(f"  Output:       {output_dir.resolve()}")


if __name__ == "__main__":
    run_demo(parse_args())
