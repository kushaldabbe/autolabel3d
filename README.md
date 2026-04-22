# autolabel3d

Open-vocabulary 3D auto-labeling for autonomous vehicle perception.

> Point it at a dashcam video or a nuScenes scene and get KITTI-format 3D bounding boxes — no manual annotation required.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     autolabel3d Pipeline                      │
├──────────────┬───────────────┬──────────────┬────────────────┤
│  Data Layer  │   Detection   │ Segmentation │    Lifting     │
│              │               │              │                │
│  NuScenes    │  Grounding    │    SAM 2.1   │  Depth         │
│  DashcamMP4  │  DINO (GDINO) │  (video      │  Anything V2   │
│              │               │   tracking)  │                │
│  ─────────── │  ─────────── │  ──────────  │  ───────────── │
│  schemas.py  │  Open-vocab   │  Temporal    │  Metric-scale  │
│  base.py     │  2D detect    │  mask prop.  │  3D box fit    │
└──────────────┴───────────────┴──────────────┴────────────────┘
          │             │               │              │
          └─────────────┴───────────────┴──────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Pipeline.run()     │
                    │  • video mode          │
                    │  • frame-by-frame mode │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        KITTI labels      Visualizations      Evaluation
        labels/*.txt      bev, overlay        mAP@0.5
```

### Two execution modes

| Mode | Detection | Segmentation | Best for |
|------|-----------|--------------|----------|
| **Video** (default) | Frame 0 only | SAM 2 propagates across all frames via memory attention | Dashcam videos, temporally consistent track IDs |
| **Frame-by-frame** | Every frame | Per-frame SAM 2 | Random single-frame inference, nuScenes |

---

## Components

| Module | Implementation | Role |
|--------|---------------|------|
| **Detector** | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Open-vocabulary 2D detection — no fixed class list |
| **Segmentor** | [SAM 2.1](https://github.com/facebookresearch/sam2) | Video-level mask propagation with temporal memory |
| **Lifter** | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Monocular depth → 3D box via PCA fitting |
| **Lifter (alt)** | Geometric Boxer | Distance from focal length + prior dimensions |
| **Optimizer** | ONNX Runtime | 1.5–3× faster depth inference (CoreML / CUDA / CPU) |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.1 ([install](https://pytorch.org/get-started/locally/))
- CUDA 11.8+ (optional, for GPU acceleration)

### Install

```bash
git clone https://github.com/kushaldabbe/autolabel3d.git
cd autolabel3d
pip install -e ".[dev]"
```

### Optional dependencies

```bash
# ONNX export + runtime
pip install onnx onnxruntime onnxsim

# nuScenes evaluation
pip install nuscenes-devkit

# Faster video encoding
brew install ffmpeg   # macOS
# apt install ffmpeg  # Ubuntu
```

---

## Quick Start

### Dashcam video

```bash
python scripts/demo.py \
    --input data/dashcam.mp4 \
    --output outputs/demo \
    --max-frames 30 \
    --every-n 5
```

This produces:
```
outputs/demo/
├── labels/             # KITTI .txt annotation files
├── visualizations/     # per-frame: detections, masks, 3D projected, BEV
└── videos/
    ├── combined_panel.mp4   # 2×2 panel: original | masks | 3D | BEV
    ├── masks_overlay.mp4
    └── boxes_3d.mp4
```

### Single image

```bash
python scripts/demo.py --input data/sample.jpg --lifter boxer
```

### Hydra CLI (full config)

```bash
# Default: nuScenes + GDINO + SAM2 + DepthAnything
python -m autolabel3d.cli

# Override data source and lifter
python -m autolabel3d.cli data=dashcam lifter=boxer

# Limit frames for quick testing
python -m autolabel3d.cli +max_frames=5

# Custom device and output
python -m autolabel3d.cli pipeline.device=cuda pipeline.output_dir=outputs/run1
```

---

## Configuration

Configs live in `configs/` and use [Hydra](https://hydra.cc/) composition.

```
configs/
├── config.yaml          ← root config (compose all modules)
├── data/
│   ├── dashcam.yaml     ← OpenCV video reader
│   └── nuscenes.yaml    ← nuScenes devkit loader
├── detector/
│   └── grounding_dino.yaml
├── lifter/
│   ├── depth_anything.yaml
│   └── boxer.yaml
└── segmentor/
    └── sam2.yaml
```

Key knobs:

```yaml
# configs/detector/grounding_dino.yaml
box_threshold: 0.3      # lower = more detections, more false positives
text_threshold: 0.25
nms:
  enabled: true
  iou_threshold: 0.5    # per-class NMS threshold
```

```yaml
# configs/lifter/depth_anything.yaml
fitting:
  method: pca           # "pca" or "min_enclosing"
  min_points: 50        # minimum mask points to attempt lifting
  height_prior:
    car: 1.5            # metres — used for metric-scale recovery
    pedestrian: 1.7
```

---

## ONNX Optimization

Export the depth model to ONNX for 1.5–3× faster inference:

```bash
# Export (downloads weights, traces graph, validates, simplifies)
python scripts/benchmark_onnx.py --model-size small

# Benchmark: PyTorch vs ONNX Runtime
python scripts/benchmark_onnx.py --skip-export --iterations 20
```

Sample output:

```
Metric          PyTorch            ONNX Runtime (CoreML)
---------       -----------------  ---------------------
Mean latency    312.4 ms           108.7 ms
Speedup         1.00× (baseline)   2.87×
```

Execution provider auto-selection: `CoreML` (Apple ANE) → `CUDA` → `CPU`.

---

## Evaluation

```python
from autolabel3d.evaluation.metrics import evaluate

# predictions and ground_truth are lists of lists of BBox3D
result = evaluate(predictions, ground_truth, iou_threshold=0.5)
print(result.summary())
# mAP@0.5: 0.412
# car AP:  0.531 | pedestrian AP: 0.312 | cyclist AP: 0.393
```

Metrics:
- **3D IoU**: exact rotated box intersection via Sutherland-Hodgman polygon clipping
- **BEV IoU**: bird's-eye view intersection (ignores height)
- **mAP**: PASCAL VOC all-point interpolation, per class then averaged

---

## Adding a New Component

1. Implement a class extending the relevant base (e.g. `BaseDetector`)
2. Add an entry to the registry in `factory.py`:
   ```python
   DETECTOR_REGISTRY["yolo"] = ("autolabel3d.detection.yolo", "YOLODetector")
   ```
3. Create a config YAML at `configs/detector/yolo.yaml`
4. Run: `python -m autolabel3d.cli detector=yolo`

---

## Project Structure

```
autolabel3d/
├── src/autolabel3d/
│   ├── data/           # DataLoader ABCs + nuScenes + dashcam loaders
│   ├── detection/      # BaseDetector + Grounding DINO
│   ├── segmentation/   # BaseSegmentor + SAM 2
│   ├── lifting/        # BaseLifter + Depth Anything V2 + Boxer
│   ├── evaluation/     # 3D IoU, mAP, KITTI format I/O
│   ├── optimization/   # ONNX export + OnnxDepthEstimator
│   ├── visualization/  # overlay, BEV, comparison panels
│   ├── utils/          # calibration, geometry, device, logging
│   ├── pipeline.py     # Pipeline orchestrator
│   ├── factory.py      # Lazy-import component registry
│   └── cli.py          # Hydra entry point
├── scripts/
│   ├── demo.py         # Standalone argparse demo
│   └── benchmark_onnx.py
├── tests/              # pytest test suite
└── configs/            # Hydra YAML configs
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_pipeline.py -v

# Lint + type check
ruff check src/
mypy src/autolabel3d/
```

