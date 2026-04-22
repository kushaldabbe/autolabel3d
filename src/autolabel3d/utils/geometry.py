"""3D geometry utilities — pinhole camera model, back-projection, bbox fitting.

THE PINHOLE CAMERA MODEL:

Full projection in homogeneous coordinates:
    s · [u, v, 1]^T = K · [R | t] · [X, Y, Z, 1]^T

where K is the 3×3 intrinsic matrix, [R|t] the extrinsic (world→camera).

BACK-PROJECTION (pixel → 3D ray):
    ray_camera = K^{-1} · [u, v, 1]^T = [(u-cx)/fx, (v-cy)/fy, 1]^T

Monocular depth is ill-posed: a pixel maps to an infinite ray.
With known depth d: P_camera = d · ray_camera.

DEPTH MAP → POINT CLOUD:
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
Applied over all pixels simultaneously (vectorised meshgrid).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from autolabel3d.data.schemas import CameraIntrinsics


def pixel_to_ray(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
) -> NDArray[np.float64]:
    """Back-project pixel coordinates to unit ray directions in camera frame.

    Applies the closed-form K^{-1} for an upper-triangular intrinsic matrix:
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1

    Args:
        u: (N,) pixel x-coordinates.
        v: (N,) pixel y-coordinates.
        intrinsics: Camera intrinsics.

    Returns:
        (N, 3) unit ray directions in camera frame.
    """
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (v - intrinsics.cy) / intrinsics.fy
    z = np.ones_like(u)

    rays = np.stack([x, y, z], axis=-1)
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays / norms


def depth_map_to_pointcloud(
    depth_map: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Convert a depth map (H, W) to a 3D point cloud in camera frame.

    Vectorised back-projection over all pixels:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Args:
        depth_map: (H, W) depth values in metres.
        intrinsics: Camera intrinsics.
        mask: Optional (H, W) boolean mask; if given, only return masked points.

    Returns:
        (N, 3) point cloud [X, Y, Z] in camera frame (invalid depths removed).
    """
    H, W = depth_map.shape

    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    X = (u_grid - intrinsics.cx) * depth_map / intrinsics.fx
    Y = (v_grid - intrinsics.cy) * depth_map / intrinsics.fy
    Z = depth_map.copy()

    points = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    points = points[mask] if mask is not None else points.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    return points[valid]


def transform_points(
    points: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply rigid body transform: P_target = R @ P_source + t.

    Args:
        points: (N, 3) points in source frame.
        rotation: (3, 3) rotation matrix.
        translation: (3,) translation vector.

    Returns:
        (N, 3) points in target frame.
    """
    return (rotation @ points.T).T + translation


def fit_3d_bbox_pca(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit a tight 3D bounding box to a point cloud using PCA.

    PCA finds the axes of maximum variance. For elongated objects (cars),
    the first eigenvector aligns with the length direction.

    Algorithm:
        1. Compute centroid; centre the points.
        2. Covariance matrix: Σ = (1/N) P'^T P'
        3. Eigendecompose (eigh for symmetric PSD matrices).
        4. Project points onto eigenvectors → compute extents.
        5. Map extents to (width, height, length).
        6. Extract yaw from first eigenvector projected onto XZ ground plane.

    Args:
        points: (N, 3) point cloud in camera frame.

    Returns:
        center:      (3,) box centre in camera frame.
        dimensions:  (3,) [width, height, length] in metres.
        rotation_y:  Yaw angle in radians (rotation around Y-axis).
    """
    if len(points) < 3:
        center = points.mean(axis=0) if len(points) > 0 else np.zeros(3)
        return center, np.array([0.1, 0.1, 0.1]), 0.0

    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)  # (3, 3) symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order — reverse for descending variance
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]

    projected = centered @ eigenvectors  # (N, 3)
    mins, maxs = projected.min(axis=0), projected.max(axis=0)
    extents = maxs - mins

    # Box centre in PCA frame → camera frame
    box_center_pca = (mins + maxs) / 2
    center = centroid + eigenvectors @ box_center_pca

    # Yaw from largest-variance axis projected onto ground plane (XZ)
    heading = eigenvectors[:, 0]
    rotation_y = float(np.arctan2(heading[0], heading[2]))

    # Assign extents to width/height/length by descending size
    sorted_ext = np.sort(extents)[::-1]
    dimensions = np.array([sorted_ext[1], sorted_ext[2], sorted_ext[0]], dtype=np.float64)

    return center, dimensions, rotation_y


def fit_3d_bbox_min_enclosing(
    points: NDArray[np.float64],
    height_prior: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Fit a 3D bounding box with a ground-plane-aligned footprint.

    More robust than PCA for AV objects: enforces that objects sit on the
    ground plane (Y = const in camera frame).

    Algorithm:
        1. Project points onto XZ ground plane.
        2. 2D PCA on XZ to find heading.
        3. Compute length/width from 2D extents.
        4. Height from Y-range (or height_prior).
        5. Reconstruct 3D centre.

    Args:
        points: (N, 3) point cloud in camera frame.
        height_prior: Override height (metres). Useful with class size priors.

    Returns:
        center:      (3,) box centre.
        dimensions:  (3,) [width, height, length] in metres.
        rotation_y:  Yaw angle in radians.
    """
    if len(points) < 3:
        center = points.mean(axis=0) if len(points) > 0 else np.zeros(3)
        return center, np.array([0.1, 0.1, 0.1]), 0.0

    xz = points[:, [0, 2]]   # (N, 2)
    y_vals = points[:, 1]     # (N,)

    centroid_2d = xz.mean(axis=0)
    centered_2d = xz - centroid_2d
    cov_2d = np.cov(centered_2d.T)

    if cov_2d.ndim < 2:
        cov_2d = np.array([[float(cov_2d), 0.0], [0.0, 1e-6]])

    evals, evecs = np.linalg.eigh(cov_2d)
    evecs = evecs[:, evals.argsort()[::-1]]

    proj_2d = centered_2d @ evecs
    mins_2d, maxs_2d = proj_2d.min(axis=0), proj_2d.max(axis=0)

    length = float(maxs_2d[0] - mins_2d[0])
    width  = float(maxs_2d[1] - mins_2d[1])

    heading_2d = evecs[:, 0]
    rotation_y = float(np.arctan2(heading_2d[0], heading_2d[1]))

    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    height = height_prior if height_prior is not None else (y_max - y_min)
    height = max(height, 0.1)

    center_2d_local = (mins_2d + maxs_2d) / 2
    center_xz = centroid_2d + evecs @ center_2d_local
    center = np.array(
        [center_xz[0], (y_min + y_max) / 2, center_xz[1]],
        dtype=np.float64,
    )
    dimensions = np.array([max(width, 0.1), height, max(length, 0.1)], dtype=np.float64)

    return center, dimensions, rotation_y
