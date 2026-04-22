"""nuScenes dataset loader.

Loads camera frames, calibration, and ground truth 3D annotations from
the nuScenes dataset. Compatible with v1.0-mini (~4 GB) for development
and v1.0-trainval for full evaluation.

nuScenes data layout:
    nuscenes/
    ├── samples/          # Keyframe images (2 Hz)
    │   ├── CAM_FRONT/
    │   └── ...           # 6 cameras total
    ├── sweeps/           # Non-keyframe images (12 Hz)
    └── v1.0-mini/        # JSON metadata
        ├── scene.json
        ├── sample.json
        ├── sample_data.json
        ├── calibrated_sensor.json
        ├── ego_pose.json
        └── sample_annotation.json

Key concepts:
    Scene:       20-second driving clip (~40 keyframes at 2 Hz)
    Sample:      One keyframe timestamp (data from all 6 cameras + LiDAR)
    Sample_data: One sensor reading (e.g., single camera image file)
    Annotation:  One labeled 3D box at a keyframe
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from autolabel3d.data.base import BaseDataLoader
from autolabel3d.data.schemas import (
    BBox3D,
    Frame,
    FrameAnnotations,
    ObjectClass,
)
from autolabel3d.utils.calibration import make_calibration_from_nuscenes
from autolabel3d.utils.logging import get_logger

logger = get_logger(__name__)

# Map nuScenes category strings → our ObjectClass enum
NUSCENES_CATEGORY_MAP: dict[str, ObjectClass] = {
    "vehicle.car":                            ObjectClass.CAR,
    "vehicle.truck":                          ObjectClass.CAR,
    "vehicle.bus.bendy":                      ObjectClass.CAR,
    "vehicle.bus.rigid":                      ObjectClass.CAR,
    "vehicle.construction":                   ObjectClass.CAR,
    "human.pedestrian.adult":                 ObjectClass.PEDESTRIAN,
    "human.pedestrian.child":                 ObjectClass.PEDESTRIAN,
    "human.pedestrian.construction_worker":   ObjectClass.PEDESTRIAN,
    "human.pedestrian.police_officer":        ObjectClass.PEDESTRIAN,
    "vehicle.bicycle":                        ObjectClass.CYCLIST,
    "vehicle.motorcycle":                     ObjectClass.CYCLIST,
    "movable_object.trafficcone":             ObjectClass.TRAFFIC_CONE,
}


class NuScenesLoader(BaseDataLoader):
    """Loads frames and ground truth annotations from the nuScenes dataset.

    The nuScenes DB is lazy-loaded on first access to avoid the 5-10 second
    JSON parsing cost at import time.

    Example:
        loader = NuScenesLoader(cfg)
        for frame in loader.load_frames():
            gt = loader.get_ground_truth(frame.frame_idx)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dataroot = Path(cfg.dataroot)
        self.cameras: list[str] = list(cfg.cameras)
        self.target_classes: set[str] = set(cfg.classes)

        # Lazy-loaded state
        self._nusc = None
        self._frame_index: list[dict] = []

    # ------------------------------------------------------------------
    # Lazy-load nuScenes DB
    # ------------------------------------------------------------------

    @property
    def nusc(self):
        """Lazy-load the nuScenes database (parses all JSON metadata on first call)."""
        if self._nusc is None:
            from nuscenes.nuscenes import NuScenes

            logger.info("Loading nuScenes %s from %s...", self.cfg.version, self.dataroot)
            self._nusc = NuScenes(
                version=self.cfg.version,
                dataroot=str(self.dataroot),
                verbose=False,
            )
            self._build_frame_index()
            logger.info(
                "Loaded %d scenes → %d frames indexed",
                len(self._nusc.scene), len(self._frame_index),
            )
        return self._nusc

    # ------------------------------------------------------------------
    # Frame index construction
    # ------------------------------------------------------------------

    def _build_frame_index(self) -> None:
        """Pre-build a flat list of (sample_token, camera_name) pairs to process.

        Respects split filtering, camera selection, and sampling strategy from config.
        """
        from nuscenes.utils import splits as nuscenes_splits

        split_scenes = getattr(nuscenes_splits, self.cfg.split, None)
        if split_scenes is None:
            raise ValueError(
                f"Unknown split '{self.cfg.split}'. "
                f"Available: mini_train, mini_val, train, val, test"
            )

        available_names = {s["name"] for s in self._nusc.scene}
        split_scenes = [s for s in split_scenes if s in available_names]

        if not split_scenes:
            raise ValueError(
                f"No scenes found for split '{self.cfg.split}' in "
                f"version '{self.cfg.version}'"
            )

        logger.info("Split '%s': %d scenes", self.cfg.split, len(split_scenes))

        self._frame_index = []
        frame_count = 0
        max_frames = self.cfg.sampling.get("max_frames", None)

        for scene in self._nusc.scene:
            if scene["name"] not in split_scenes:
                continue

            sample_token = scene["first_sample_token"]
            sample_idx = 0

            while sample_token:
                if max_frames and frame_count >= max_frames:
                    break

                sample = self._nusc.get("sample", sample_token)

                if self._should_include(sample_idx):
                    for cam in self.cameras:
                        if cam in sample["data"]:
                            self._frame_index.append({
                                "sample_token": sample_token,
                                "camera_name": cam,
                                "camera_token": sample["data"][cam],
                                "scene_name": scene["name"],
                                "timestamp": sample["timestamp"] / 1e6,
                            })
                            frame_count += 1

                sample_token = sample["next"]
                sample_idx += 1

            if max_frames and frame_count >= max_frames:
                break

    def _should_include(self, sample_idx: int) -> bool:
        strategy = self.cfg.sampling.strategy
        if strategy in ("keyframe", "all"):
            return True
        if strategy == "every_n":
            return sample_idx % self.cfg.sampling.every_n == 0
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    # ------------------------------------------------------------------
    # BaseDataLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        _ = self.nusc  # trigger lazy load
        return len(self._frame_index)

    def load_frames(self) -> Iterator[Frame]:
        for idx in range(len(self)):
            yield self.get_frame(idx)

    def get_frame(self, idx: int) -> Frame:
        """Load a specific frame by index: read image + extract calibration."""
        entry = self._frame_index[idx]
        nusc = self.nusc

        sd_record = nusc.get("sample_data", entry["camera_token"])
        img_path = self.dataroot / sd_record["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        ego_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
        K = np.array(cs_record["camera_intrinsic"])

        calibration = make_calibration_from_nuscenes(K, cs_record, ego_record)

        return Frame(
            image=image,
            frame_idx=idx,
            timestamp=entry["timestamp"],
            camera_name=entry["camera_name"],
            calibration=calibration,
            source_path=img_path,
        )

    # ------------------------------------------------------------------
    # Ground truth access
    # ------------------------------------------------------------------

    def get_ground_truth(self, frame_idx: int) -> FrameAnnotations:
        """Return ground truth 3D boxes for a frame (for evaluation).

        Loads nuScenes annotations and converts them to our BBox3D format.
        """
        from pyquaternion import Quaternion

        entry = self._frame_index[frame_idx]
        sample = self.nusc.get("sample", entry["sample_token"])

        boxes_3d: list[BBox3D] = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            obj_class = NUSCENES_CATEGORY_MAP.get(ann["category_name"])
            if obj_class is None or obj_class.value not in self.target_classes:
                continue

            # nuScenes size order: (width, length, height) → we want (width, height, length)
            w, l, h = ann["size"]
            rotation_y = float(Quaternion(ann["rotation"]).yaw_pitch_roll[0])

            boxes_3d.append(BBox3D(
                center=np.array(ann["translation"], dtype=np.float64),
                dimensions=np.array([w, h, l], dtype=np.float64),
                rotation_y=rotation_y,
                class_name=obj_class,
                confidence=1.0,
                track_id=hash(ann["instance_token"]) % (10 ** 6),
            ))

        return FrameAnnotations(frame_idx=frame_idx, boxes_3d=boxes_3d)

    @staticmethod
    def _map_category(category_name: str) -> ObjectClass | None:
        return NUSCENES_CATEGORY_MAP.get(category_name)
