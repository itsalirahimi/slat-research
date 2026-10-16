"""Image and trajectory diagnostics for mapping checkpoints."""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # Saving maps already requires OpenCV through MapIO.
    cv2 = None

from scipy.spatial import cKDTree

from .products import RgbMap2D


def visual_diagnostic_images(
    source_bgr: Optional[np.ndarray],
    target_bgr: Optional[np.ndarray],
    diagnostic: Optional[Dict],
    max_matches: int = 250,
    max_dimension: int = 1_600,
) -> Dict[str, np.ndarray]:
    """Draw mutual matches, RGB registration overlay, and depth-valid masks."""
    if cv2 is None or source_bgr is None or target_bgr is None or not diagnostic:
        return {}
    source = np.asarray(source_bgr, dtype=np.uint8)
    target = np.asarray(target_bgr, dtype=np.uint8)

    def resize_for_report(image):
        scale = min(1.0, float(max_dimension) / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        return image, scale

    source, source_scale = resize_for_report(source)
    target, target_scale = resize_for_report(target)
    source_xy = np.asarray(diagnostic.get("source_xy", []), dtype=np.float64).reshape((-1, 2))
    target_xy = np.asarray(diagnostic.get("target_xy", []), dtype=np.float64).reshape((-1, 2))
    source_xy *= source_scale
    target_xy *= target_scale
    inliers = np.asarray(diagnostic.get("inlier_mask", []), dtype=bool).reshape(-1)
    if len(inliers) != len(source_xy):
        inliers = np.zeros(len(source_xy), dtype=bool)

    height = max(source.shape[0], target.shape[0])
    matches_image = np.zeros((height, source.shape[1] + target.shape[1], 3), dtype=np.uint8)
    matches_image[: source.shape[0], : source.shape[1]] = source
    matches_image[: target.shape[0], source.shape[1] :] = target
    if len(source_xy):
        count = min(len(source_xy), max(1, int(max_matches)))
        selected = np.linspace(0, len(source_xy) - 1, count, dtype=np.int64)
        for index in selected:
            left = tuple(np.rint(source_xy[index]).astype(int))
            right_values = np.rint(target_xy[index]).astype(int)
            right = (int(right_values[0] + source.shape[1]), int(right_values[1]))
            color = (70, 220, 70) if inliers[index] else (80, 80, 245)
            cv2.line(matches_image, left, right, color, 1, cv2.LINE_AA)
            cv2.circle(matches_image, left, 2, color, -1, cv2.LINE_AA)
            cv2.circle(matches_image, right, 2, color, -1, cv2.LINE_AA)
    cv2.putText(
        matches_image,
        "current/source",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        matches_image,
        "keyframe/target",
        (source.shape[1] + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    homography = diagnostic.get("homography")
    if homography is None:
        overlay = target.copy()
        mask_overlay = np.zeros(target.shape[:2] + (3,), dtype=np.uint8)
    else:
        homography = np.asarray(homography, dtype=np.float64)
        source_scaling = np.diag([source_scale, source_scale, 1.0])
        target_scaling = np.diag([target_scale, target_scale, 1.0])
        homography = target_scaling @ homography @ np.linalg.inv(source_scaling)
        target_size = (target.shape[1], target.shape[0])
        warped = cv2.warpPerspective(source, homography, target_size)

        def coverage_mask(shape, points):
            mask = np.zeros(shape, dtype=np.uint8)
            if len(points) >= 3:
                hull = cv2.convexHull(np.rint(points).astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 255)
            else:
                for point in points:
                    cv2.circle(mask, tuple(np.rint(point).astype(int)), 8, 255, -1)
            return mask

        kept_source = source_xy[inliers] if np.any(inliers) else source_xy
        kept_target = target_xy[inliers] if np.any(inliers) else target_xy
        source_mask = coverage_mask(source.shape[:2], kept_source)
        target_mask = coverage_mask(target.shape[:2], kept_target)
        warped_mask = cv2.warpPerspective(
            source_mask, homography, target_size, flags=cv2.INTER_NEAREST
        ) > 0
        target_mask = target_mask > 0
        overlap_mask = warped_mask & target_mask
        overlay = target.copy()
        overlay[warped_mask & ~target_mask] = warped[warped_mask & ~target_mask]
        overlay[overlap_mask] = np.rint(
            0.5 * target[overlap_mask].astype(np.float32)
            + 0.5 * warped[overlap_mask].astype(np.float32)
        ).astype(np.uint8)
        mask_overlay = np.zeros(target.shape[:2] + (3,), dtype=np.uint8)
        mask_overlay[target_mask] = (255, 0, 255)  # Magenta: target only.
        mask_overlay[warped_mask] = (255, 255, 0)  # Cyan: source only.
        mask_overlay[overlap_mask] = (255, 255, 255)

    return {
        "matches": matches_image,
        "rgb_overlay": overlay,
        "mask_overlay": mask_overlay,
    }


def geometry_error_cloud(
    source_registered: np.ndarray,
    target: np.ndarray,
    threshold: float,
    max_points: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Create a colored source/target alignment cloud and distance statistics."""
    source = np.asarray(source_registered, dtype=np.float64).reshape((-1, 3))
    target = np.asarray(target, dtype=np.float64).reshape((-1, 3))
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if len(source) > max_points // 2:
        source = source[np.linspace(0, len(source) - 1, max_points // 2, dtype=np.int64)]
    if len(target) > max_points // 2:
        target = target[np.linspace(0, len(target) - 1, max_points // 2, dtype=np.int64)]
    if not len(source) or not len(target):
        return np.empty((0, 3)), np.empty((0, 3)), {"samples": 0}

    distances, _ = cKDTree(target).query(source, workers=-1)
    scale = max(float(threshold), 1e-9)
    normalized = np.clip(distances / scale, 0.0, 1.0)
    source_colors = np.column_stack(
        (normalized, 1.0 - normalized, np.full(len(source), 0.12))
    )
    target_colors = np.tile([0.95, 0.68, 0.35], (len(target), 1))
    points = np.vstack((target, source))
    colors = np.vstack((target_colors, source_colors))
    report = {
        "samples": int(len(source)),
        "target_samples": int(len(target)),
        "within_threshold": float(np.mean(distances <= scale)),
        "rmse_all": float(np.sqrt(np.mean(distances ** 2))),
        "median_all": float(np.median(distances)),
        "p95_all": float(np.quantile(distances, 0.95)),
        "color_key": {
            "target": "light orange",
            "registered_source_near": "green",
            "registered_source_far": "red",
        },
    }
    return points, colors, report


def trajectory_image(
    rgb_map: RgbMap2D,
    camera_positions: np.ndarray,
    accepted: Iterable[bool],
) -> np.ndarray:
    """Draw accepted/rejected camera centers over the orthographic RGB map."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required to draw the 2D trajectory")
    image = cv2.cvtColor(rgb_map.image_rgb, cv2.COLOR_RGB2BGR)
    # Make unseen map pixels dark gray so the colored trajectory stays visible.
    image[rgb_map.mask == 0] = (35, 35, 35)
    positions = np.asarray(camera_positions, dtype=np.float64).reshape((-1, 3))
    if not len(positions):
        return image
    pixels = rgb_map.world_to_pixel(positions[:, :2])
    accepted = np.asarray(list(accepted), dtype=bool)
    for index in range(1, len(pixels)):
        if accepted[index - 1] and accepted[index]:
            cv2.line(
                image,
                tuple(pixels[index - 1]),
                tuple(pixels[index]),
                (255, 220, 30),
                2,
                cv2.LINE_AA,
            )
    for index, pixel in enumerate(pixels):
        color = (40, 240, 40) if accepted[index] else (40, 40, 245)
        cv2.circle(image, tuple(pixel), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (int(pixel[0]) + 5, int(pixel[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image
