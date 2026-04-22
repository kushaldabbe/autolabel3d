#!/usr/bin/env python3
"""Benchmark ONNX Runtime vs PyTorch for Depth Anything V2.

Usage:
    # Export then benchmark
    python scripts/benchmark_onnx.py

    # Benchmark only (model already exported)
    python scripts/benchmark_onnx.py --skip-export

    # Custom settings
    python scripts/benchmark_onnx.py --model-size small --warmup 3 --iterations 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ONNX vs PyTorch depth estimation")
    p.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    p.add_argument("--input", default=None, help="Image or video to use (default: synthetic)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--input-size", type=int, default=518)
    return p.parse_args()


def get_test_image(input_path: str | None, size: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    if input_path:
        path = Path(input_path)
        if path.suffix.lower() in {".mp4", ".avi", ".mov"}:
            cap = cv2.VideoCapture(str(path))
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
        else:
            img = cv2.imread(str(path))
            if img is not None:
                return img
    return np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)


def benchmark_pytorch(
    model_size: str, image: np.ndarray, warmup: int, iterations: int,
) -> dict:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from autolabel3d.utils.device import get_device

    device = get_device()
    model_id = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"

    print(f"\n  Loading PyTorch model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = {k: v.to(device) for k, v in processor(images=rgb, return_tensors="pt").items()}

    print(f"  Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(**inputs)
        if device.type == "mps":
            torch.mps.synchronize()

    print(f"  Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs)
        if device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "backend": f"PyTorch ({device})",
        "times": times,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "output_shape": tuple(out.predicted_depth.shape),
    }


def benchmark_onnx(
    onnx_path: Path, image: np.ndarray, warmup: int, iterations: int,
) -> dict:
    from autolabel3d.optimization.onnx_runtime import OnnxDepthEstimator

    estimator = OnnxDepthEstimator(onnx_path)
    provider = estimator.session.get_providers()[0]

    print(f"  Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        _ = estimator.predict(image)

    print(f"  Benchmarking ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        depth = estimator.predict(image)
        times.append(time.perf_counter() - t0)

    return {
        "backend": f"ONNX Runtime ({provider})",
        "times": times,
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "output_shape": depth.shape,
    }


def print_results(pt: dict, ort: dict) -> None:
    speedup = pt["mean_ms"] / max(ort["mean_ms"], 0.01)
    rows = [
        ("Backend",      pt["backend"],               ort["backend"]),
        ("Mean latency", f"{pt['mean_ms']:.1f} ms",   f"{ort['mean_ms']:.1f} ms"),
        ("Std dev",      f"{pt['std_ms']:.1f} ms",    f"{ort['std_ms']:.1f} ms"),
        ("Min latency",  f"{pt['min_ms']:.1f} ms",    f"{ort['min_ms']:.1f} ms"),
        ("Output shape", str(pt["output_shape"]),      str(ort["output_shape"])),
        ("Speedup",      "1.00× (baseline)",           f"{speedup:.2f}×"),
    ]
    header = ("Metric", "PyTorch", "ONNX Runtime")
    all_rows = [header, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(3)]
    fmt = "  {:<{w0}}  {:<{w1}}  {:<{w2}}"

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(fmt.format(*header, w0=widths[0], w1=widths[1], w2=widths[2]))
    print("  " + "-" * (sum(widths) + 4))
    for row in rows:
        print(fmt.format(*row, w0=widths[0], w1=widths[1], w2=widths[2]))
    print(f"\n  ONNX Runtime is {speedup:.1f}× {'faster' if speedup > 1 else 'slower'} than PyTorch")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    onnx_path = Path(f"weights/depth_anything_v2_{args.model_size}.onnx")

    print("=" * 60)
    print("Depth Anything V2 — PyTorch vs ONNX Runtime Benchmark")
    print("=" * 60)

    if not args.skip_export or not onnx_path.exists():
        print("\n[1/3] Exporting to ONNX...")
        from autolabel3d.optimization.export import export_depth_anything
        onnx_path = export_depth_anything(
            model_size=args.model_size, output_path=onnx_path,
            input_height=args.input_size, input_width=args.input_size,
        )
    else:
        print(f"\n[1/3] Using existing ONNX model: {onnx_path}")

    image = get_test_image(args.input)
    print(f"\n[2/3] Test image: {image.shape[1]}×{image.shape[0]}")

    print(f"\n[3/3] Running benchmarks ({args.warmup} warmup + {args.iterations} timed)...")
    print("\n--- PyTorch ---")
    pt_results = benchmark_pytorch(args.model_size, image, args.warmup, args.iterations)
    print("\n--- ONNX Runtime ---")
    ort_results = benchmark_onnx(onnx_path, image, args.warmup, args.iterations)
    print_results(pt_results, ort_results)


if __name__ == "__main__":
    main()
