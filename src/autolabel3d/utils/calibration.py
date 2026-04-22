"""Camera calibration utilities — nuScenes format converters.

nuScenes stores calibration as a chain:
    world → ego vehicle → camera sensor → image pixel

We compose these into a single camera→world transform for 3D lifting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyquaternion import Quaternion

from autolabel3d.data.schemas import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)


def intrinsics_from_matrix(K: NDArray[np.float64]) -> CameraIntrinsics:
    """Extract CameraIntrinsics from a 3×3 intrinsic matrix K."""
    return CameraIntrinsics(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )


def extrinsics_from_nuscenes(
    sensor_record: dict,
    ego_pose_record: dict,
) -> CameraExtrinsics:
    """Build CameraExtrinsics from nuScenes calibrated_sensor + ego_pose records.

    Composes the full camera→world transform:
        T_world_camera = T_world_ego @ T_ego_camera

    Args:
        sensor_record: nuScenes 'calibrated_sensor' record.
        ego_pose_record: nuScenes 'ego_pose' record.

    Returns:
        CameraExtrinsics representing the camera→world rigid transform.
    """
    # Sensor (camera) → ego vehicle
    T_ego_sensor = np.eye(4)
    T_ego_sensor[:3, :3] = Quaternion(sensor_record["rotation"]).rotation_matrix
    T_ego_sensor[:3, 3] = np.array(sensor_record["translation"])

    # Ego vehicle → world
    T_world_ego = np.eye(4)
    T_world_ego[:3, :3] = Quaternion(ego_pose_record["rotation"]).rotation_matrix
    T_world_ego[:3, 3] = np.array(ego_pose_record["translation"])

    T_world_camera = T_world_ego @ T_ego_sensor

    return CameraExtrinsics(
        rotation=T_world_camera[:3, :3].astype(np.float64),
        translation=T_world_camera[:3, 3].astype(np.float64),
    )


def make_calibration_from_nuscenes(
    camera_intrinsic: NDArray[np.float64],
    sensor_record: dict,
    ego_pose_record: dict,
) -> CameraCalibration:
    """Build a full CameraCalibration from nuScenes metadata records."""
    return CameraCalibration(
        intrinsics=intrinsics_from_matrix(camera_intrinsic),
        extrinsics=extrinsics_from_nuscenes(sensor_record, ego_pose_record),
    )
