"""Visual, ground-plane, and geometric registration helpers."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:  # Keep pure-numpy helpers importable in lightweight test environments.
    import cv2
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    cv2 = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    o3d = None

from .types import FeatureSet


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def make_translation(translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper-rigid transform mapping source onto target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def transform_difference(
    first: np.ndarray,
    second: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Position and rotation separating two source-to-target transforms.

    Supplying a reference point avoids exaggerating translation when two
    transforms express a small rotation about a distant coordinate origin.
    """
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    relative_rotation = first[:3, :3] @ second[:3, :3].T
    if reference_point is None:
        delta = first @ np.linalg.inv(second)
        distance = float(np.linalg.norm(delta[:3, 3]))
    else:
        reference = np.asarray(reference_point, dtype=np.float64).reshape(1, 3)
        distance = float(
            np.linalg.norm(
                transform_points(reference, first) - transform_points(reference, second)
            )
        )
    return distance, rotation_angle_deg(relative_rotation)


def ransac_rigid(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[float]]:
    """Robust SE(3) estimation for known 3D correspondences."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_points = len(source)
    empty_mask = np.zeros(n_points, dtype=bool)
    if n_points < 3:
        return None, empty_mask, None

    best_transform = None
    best_mask = empty_mask
    best_score = (-1, -np.inf)

    for _ in range(max(1, int(iterations))):
        sample = rng.choice(n_points, size=3, replace=False)
        if (
            np.linalg.matrix_rank(source[sample] - source[sample].mean(axis=0)) < 2
            or np.linalg.matrix_rank(target[sample] - target[sample].mean(axis=0)) < 2
        ):
            continue
        try:
            transform = rigid_transform(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        mask = residuals <= threshold
        count = int(mask.sum())
        median = float(np.median(residuals[mask])) if count else np.inf
        score = (count, -median)
        if score > best_score:
            best_transform, best_mask, best_score = transform, mask, score
            if count == n_points:
                break

    if best_transform is None or int(best_mask.sum()) < 3:
        return None, empty_mask, None

    # Refit and reclassify twice to remove the bias of the minimal sample.
    for _ in range(2):
        best_transform = rigid_transform(source[best_mask], target[best_mask])
        residuals = np.linalg.norm(
            transform_points(source, best_transform) - target, axis=1
        )
        best_mask = residuals <= threshold
        if int(best_mask.sum()) < 3:
            return None, empty_mask, None
    rmse = float(np.sqrt(np.mean(residuals[best_mask] ** 2)))
    return best_transform, best_mask, rmse


def extract_sift_features(
    image: Optional[np.ndarray], max_features: int, max_dimension: int
) -> Optional[FeatureSet]:
    if image is None or cv2 is None:
        return None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale < 1.0:
        work = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = gray

    detector = cv2.SIFT_create(nfeatures=int(max_features))
    keypoints, descriptors = detector.detectAndCompute(work, None)
    if not keypoints or descriptors is None:
        return FeatureSet(
            points_xy=np.empty((0, 2), dtype=np.float64),
            descriptors=None,
            image_shape=(height, width),
        )
    points_xy = np.asarray([kp.pt for kp in keypoints], dtype=np.float64) / scale
    return FeatureSet(points_xy, descriptors, (height, width))


def _ratio_matches(query: np.ndarray, train: np.ndarray, ratio: float):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(query, train, k=2)
    accepted = {}
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            accepted[pair[0].queryIdx] = pair[0].trainIdx
    return accepted


def match_sift_features(
    source: FeatureSet,
    target: FeatureSet,
    ratio: float,
    ransac_px: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "source_indices": np.empty(0, dtype=np.int64),
        "target_indices": np.empty(0, dtype=np.int64),
        "matches": 0,
        "ransac_2d_inliers": 0,
        "model": "none",
    }
    if cv2 is None or source.descriptors is None or target.descriptors is None:
        return result
    if len(source.descriptors) < 2 or len(target.descriptors) < 2:
        return result

    forward = _ratio_matches(source.descriptors, target.descriptors, ratio)
    reverse = _ratio_matches(target.descriptors, source.descriptors, ratio)
    mutual = sorted((q, t) for q, t in forward.items() if reverse.get(t) == q)
    result["matches"] = len(mutual)
    if len(mutual) < 4:
        return result

    source_indices = np.asarray([item[0] for item in mutual], dtype=np.int64)
    target_indices = np.asarray([item[1] for item in mutual], dtype=np.int64)
    source_xy = source.points_xy[source_indices]
    target_xy = target.points_xy[target_indices]

    candidates = []
    if len(mutual) >= 8:
        method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
        try:
            _, mask = cv2.findFundamentalMat(
                source_xy, target_xy, method, ransac_px, 0.999
            )
            if mask is not None and mask.size == len(mutual):
                candidates.append(("fundamental", mask.reshape(-1).astype(bool)))
        except cv2.error:
            pass
    try:
        _, mask = cv2.findHomography(
            source_xy, target_xy, cv2.RANSAC, ransac_px, confidence=0.999
        )
        if mask is not None and mask.size == len(mutual):
            candidates.append(("homography", mask.reshape(-1).astype(bool)))
    except cv2.error:
        pass

    if not candidates:
        return result
    model, keep = max(candidates, key=lambda item: int(item[1].sum()))
    result.update(
        source_indices=source_indices[keep],
        target_indices=target_indices[keep],
        ransac_2d_inliers=int(keep.sum()),
        model=model,
    )
    return result


def _pixel_indices(points_xy: np.ndarray, image_shape: Tuple[int, int]):
    height, width = image_shape
    u = np.rint(points_xy[:, 0]).astype(np.int64)
    v = np.rint(points_xy[:, 1]).astype(np.int64)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return v * width + u, valid


def _coverage(points_xy: np.ndarray, image_shape: Tuple[int, int]) -> float:
    if len(points_xy) < 2:
        return 0.0
    extent = np.ptp(points_xy, axis=0)
    height, width = image_shape
    return float((extent[0] * extent[1]) / max(float(width * height), 1.0))


def visual_registration(
    source_features: Optional[FeatureSet],
    source_ordered_points: np.ndarray,
    source_initial_transform: np.ndarray,
    target_features: Optional[FeatureSet],
    target_ordered_points: np.ndarray,
    target_transform: np.ndarray,
    ratio: float,
    ransac_px: float,
    threshold_3d: float,
    ransac_iterations: int,
    min_inliers: int,
    min_coverage: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "success": False,
        "reason": "features_unavailable",
        "matches": 0,
        "ransac_2d_inliers": 0,
        "lifted_3d": 0,
        "inliers_3d": 0,
        "coverage": 0.0,
        "rmse_3d": None,
        "model": "none",
        "transform": np.eye(4, dtype=np.float64),
    }
    if source_features is None or target_features is None:
        return report
    if len(source_ordered_points) != np.prod(source_features.image_shape):
        report["reason"] = "source_cloud_not_image_organized"
        return report
    if len(target_ordered_points) != np.prod(target_features.image_shape):
        report["reason"] = "target_cloud_not_image_organized"
        return report

    matches = match_sift_features(source_features, target_features, ratio, ransac_px)
    report.update(
        matches=matches["matches"],
        ransac_2d_inliers=matches["ransac_2d_inliers"],
        model=matches["model"],
    )
    source_feature_indices = matches["source_indices"]
    target_feature_indices = matches["target_indices"]
    if len(source_feature_indices) < 3:
        report["reason"] = "too_few_2d_inliers"
        return report

    source_xy = source_features.points_xy[source_feature_indices]
    target_xy = target_features.points_xy[target_feature_indices]
    source_point_indices, source_valid = _pixel_indices(
        source_xy, source_features.image_shape
    )
    target_point_indices, target_valid = _pixel_indices(
        target_xy, target_features.image_shape
    )
    valid = source_valid & target_valid
    source_3d = np.asarray(source_ordered_points)[source_point_indices[valid]]
    target_3d = np.asarray(target_ordered_points)[target_point_indices[valid]]
    finite = np.isfinite(source_3d).all(axis=1) & np.isfinite(target_3d).all(axis=1)
    source_3d = transform_points(source_3d[finite], source_initial_transform)
    target_3d = transform_points(target_3d[finite], target_transform)
    source_xy = source_xy[valid][finite]
    target_xy = target_xy[valid][finite]
    report["lifted_3d"] = len(source_3d)
    if len(source_3d) < 3:
        report["reason"] = "too_few_valid_3d_matches"
        return report

    transform, inliers, rmse = ransac_rigid(
        source_3d, target_3d, threshold_3d, ransac_iterations, rng
    )
    if transform is None:
        report["reason"] = "3d_ransac_failed"
        return report

    coverage = min(
        _coverage(source_xy[inliers], source_features.image_shape),
        _coverage(target_xy[inliers], target_features.image_shape),
    )
    report.update(
        transform=transform,
        inliers_3d=int(inliers.sum()),
        coverage=coverage,
        rmse_3d=rmse,
    )
    if int(inliers.sum()) < int(min_inliers):
        report["reason"] = "too_few_3d_inliers"
    elif coverage < float(min_coverage):
        report["reason"] = "inliers_too_concentrated"
    else:
        report["success"] = True
        report["reason"] = "ok"
    return report


def nearest_neighbor_quality(
    source: np.ndarray,
    target: np.ndarray,
    threshold: float,
    max_points: int = 80_000,
) -> Dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if not len(source) or not len(target):
        return {
            "overlap": 0.0,
            "rmse": None,
            "median": None,
            "samples": 0,
        }
    if len(source) > max_points:
        sample = np.linspace(0, len(source) - 1, max_points, dtype=np.int64)
        source = source[sample]
    distances, _ = cKDTree(target).query(source, k=1, workers=-1)
    inliers = distances <= threshold
    return {
        "overlap": float(np.mean(inliers)),
        "rmse": float(np.sqrt(np.mean(distances[inliers] ** 2))) if np.any(inliers) else None,
        "median": float(np.median(distances[inliers])) if np.any(inliers) else None,
        "samples": int(len(source)),
    }


def _fit_plane(points: np.ndarray, max_points: int = 20_000):
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    keep = np.ones(len(points), dtype=bool)
    normal = None
    center = None
    for _ in range(3):
        selected = points[keep]
        if len(selected) < 3:
            return None
        center = selected.mean(axis=0)
        _, _, vt = np.linalg.svd(selected - center, full_matrices=False)
        normal = vt[-1]
        normal /= max(np.linalg.norm(normal), 1e-12)
        residuals = np.abs((points - center) @ normal)
        cutoff = max(float(np.quantile(residuals, 0.80)), 1e-4)
        keep = residuals <= cutoff
    if normal[2] < 0:
        normal = -normal
    offset = -float(normal @ center)
    rmse = float(np.sqrt(np.mean(((points[keep] @ normal) + offset) ** 2)))
    return normal, offset, center, rmse, int(keep.sum())


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-12:
        return np.eye(3, dtype=np.float64)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def ground_plane_alignment(
    source_ground_prior: np.ndarray,
    target_ground: np.ndarray,
    initial_transform: np.ndarray,
    min_points: int,
    max_tilt_deg: float,
    max_shift: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "used": False,
        "reason": "insufficient_ground",
        "tilt_deg": None,
        "normal_shift": None,
    }
    source_ground_prior = np.asarray(source_ground_prior, dtype=np.float64)
    target_ground = np.asarray(target_ground, dtype=np.float64)
    if len(source_ground_prior) < min_points or len(target_ground) < min_points:
        return initial_transform, report

    source_transformed = transform_points(source_ground_prior, initial_transform)
    source_plane = _fit_plane(source_transformed)
    target_plane = _fit_plane(target_ground)
    if source_plane is None or target_plane is None:
        report["reason"] = "plane_fit_failed"
        return initial_transform, report

    source_normal, _, source_center, source_rmse, source_count = source_plane
    target_normal, target_offset, _, target_rmse, target_count = target_plane
    rotation = _rotation_between(source_normal, target_normal)
    tilt_deg = rotation_angle_deg(rotation)
    # Rotate about the source plane center, then move only along the target
    # normal.  GEP deliberately does not invent yaw or in-plane translation.
    signed_distance = float(target_normal @ source_center + target_offset)
    normal_shift = -signed_distance
    report.update(
        tilt_deg=tilt_deg,
        normal_shift=normal_shift,
        source_plane_rmse=source_rmse,
        target_plane_rmse=target_rmse,
        source_points=source_count,
        target_points=target_count,
    )
    if tilt_deg > max_tilt_deg:
        report["reason"] = "tilt_exceeds_limit"
        return initial_transform, report
    if abs(normal_shift) > max_shift:
        report["reason"] = "height_shift_exceeds_limit"
        return initial_transform, report

    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = rotation
    correction[:3, 3] = (
        source_center - rotation @ source_center + normal_shift * target_normal
    )
    report["used"] = True
    report["reason"] = "ok"
    return correction @ initial_transform, report


def _make_cloud(points: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def icp_registration(
    source_prior: np.ndarray,
    target: np.ndarray,
    initial_transform: np.ndarray,
    voxel_size: float,
    max_points: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Robust coarse-to-fine point-to-plane ICP.

    ``source_prior`` is already positioned by the selected sensor prior;
    therefore the returned transform is the residual from that prior.
    """
    report: Dict[str, Any] = {"success": False, "reason": "dependency_missing", "levels": []}
    if o3d is None:
        return initial_transform, report

    source_prior = np.asarray(source_prior, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_prior = source_prior[np.isfinite(source_prior).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if len(source_prior) < 20 or len(target) < 20:
        report["reason"] = "insufficient_geometry"
        return initial_transform, report
    if len(source_prior) > max_points * 2:
        sample = np.linspace(0, len(source_prior) - 1, max_points * 2, dtype=np.int64)
        source_prior = source_prior[sample]

    source_full = _make_cloud(source_prior)
    target_full = _make_cloud(target)
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    registration = o3d.pipelines.registration
    levels = [
        (2.0 * voxel_size, 4.0 * voxel_size, 40),
        (voxel_size, 2.0 * voxel_size, 30),
        (0.5 * voxel_size, voxel_size, 25),
    ]

    for downsample, max_correspondence, iterations in levels:
        source = source_full.voxel_down_sample(max(downsample, 1e-4))
        target_level = target_full.voxel_down_sample(max(downsample, 1e-4))
        if len(source.points) < 20 or len(target_level.points) < 20:
            continue
        target_level.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(3.0 * downsample, max_correspondence), max_nn=50
            )
        )
        criteria = registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=iterations,
        )
        try:
            loss = registration.TukeyLoss(k=max(max_correspondence * 0.5, 1e-4))
            estimator = registration.TransformationEstimationPointToPlane(loss)
        except (AttributeError, TypeError):
            estimator = registration.TransformationEstimationPointToPlane()
        try:
            result = registration.registration_icp(
                source,
                target_level,
                max_correspondence,
                transform,
                estimator,
                criteria,
            )
        except RuntimeError:
            result = registration.registration_icp(
                source,
                target_level,
                max_correspondence,
                transform,
                registration.TransformationEstimationPointToPoint(False),
                criteria,
            )
        transform = np.asarray(result.transformation, dtype=np.float64)
        report["levels"].append(
            {
                "voxel_size": float(downsample),
                "max_correspondence": float(max_correspondence),
                "fitness": float(result.fitness),
                "rmse": float(result.inlier_rmse),
                "source_points": int(len(source.points)),
                "target_points": int(len(target_level.points)),
            }
        )

    if report["levels"]:
        report["success"] = True
        report["reason"] = "ok"
    else:
        report["reason"] = "insufficient_downsampled_geometry"
    return transform, report
