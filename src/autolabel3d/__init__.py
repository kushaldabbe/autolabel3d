"""autolabel3d — auto-labeling pipeline for autonomous vehicle perception.

End-to-end pipeline: RGB frames → 2D detections → segmentation masks → 3D bounding boxes.

Modules:
    data        — data loaders (nuScenes, dashcam video)
    detection   — Grounding DINO open-vocabulary 2D detector
    segmentation — SAM 2.1 video segmentor with temporal tracking
    lifting     — 2D→3D lifting (Depth Anything V2, geometric Boxer)
    evaluation  — 3D IoU, KITTI format I/O, mAP metrics
    optimization — ONNX export and runtime for deployment
    visualization — BEV plots, mask overlays, comparison views
    utils       — calibration, geometry, device selection, logging
"""

__version__ = "0.1.0"
__author__ = "Kushal Dabbe"
