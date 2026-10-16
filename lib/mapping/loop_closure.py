"""Optional non-destructive loop closure and global SE(3) pose optimization."""

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .registration import (
    icp_registration,
    nearest_neighbor_quality,
    rotation_angle_deg,
    transform_difference,
    transform_points,
    visual_registration,
)
from .types import Keyframe


def _relative_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the transform from the second local frame into the first."""
    return np.linalg.inv(first) @ second


def _pack_pose(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3])
    )


def _unpack_pose(values: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    transform[:3, 3] = values[3:6]
    return transform


def _edge_error(first: np.ndarray, second: np.ndarray, measurement: np.ndarray):
    error = np.linalg.inv(measurement) @ _relative_pose(first, second)
    translation = np.asarray(error[:3, 3], dtype=np.float64)
    rotation = Rotation.from_matrix(error[:3, :3]).as_rotvec()
    return translation, rotation


def _graph_metrics(poses: List[np.ndarray], edges: List[Dict]) -> Dict:
    selected = [edge for edge in edges if edge["kind"] == "loop"]
    if not selected:
        return {
            "loop_edges": 0,
            "translation_rmse_m": None,
            "rotation_rmse_deg": None,
            "translation_max_m": None,
            "rotation_max_deg": None,
        }
    translations = []
    rotations = []
    for edge in selected:
        translation, rotation = _edge_error(
            poses[edge["source"]], poses[edge["target"]], edge["measurement"]
        )
        translations.append(float(np.linalg.norm(translation)))
        rotations.append(float(np.degrees(np.linalg.norm(rotation))))
    translations = np.asarray(translations)
    rotations = np.asarray(rotations)
    return {
        "loop_edges": int(len(selected)),
        "translation_rmse_m": float(np.sqrt(np.mean(translations ** 2))),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotations ** 2))),
        "translation_max_m": float(np.max(translations)),
        "rotation_max_deg": float(np.max(rotations)),
    }


def _candidate_pairs(keyframes: List[Keyframe], config) -> List[Tuple[int, int, float]]:
    candidates = []
    for first in range(len(keyframes)):
        for second in range(first + config.loop_min_separation, len(keyframes)):
            distance = float(
                np.linalg.norm(
                    keyframes[first].transform[:3, 3]
                    - keyframes[second].transform[:3, 3]
                )
            )
            if distance <= config.loop_search_radius:
                candidates.append((first, second, distance))
    candidates.sort(key=lambda item: (item[2], item[0], item[1]))
    return candidates[: int(config.loop_max_candidates)]


def build_pose_graph(
    keyframes: List[Keyframe], config, rng: np.random.Generator
) -> Tuple[List[Dict], List[Dict]]:
    """Build odometry edges and independently verified non-adjacent edges."""
    edges: List[Dict] = []
    candidate_reports: List[Dict] = []
    for index in range(len(keyframes) - 1):
        edges.append(
            {
                "source": index,
                "target": index + 1,
                "kind": "odometry",
                "measurement": _relative_pose(
                    keyframes[index].transform, keyframes[index + 1].transform
                ),
                "weight": float(
                    np.clip(
                        0.5
                        + 0.5
                        * min(keyframes[index].confidence, keyframes[index + 1].confidence),
                        0.5,
                        1.0,
                    )
                ),
            }
        )

    for first, second, camera_distance in _candidate_pairs(keyframes, config):
        target = keyframes[first]
        source = keyframes[second]
        report: Dict = {
            "reference_pose": first,
            "query_pose": second,
            "reference_frame_index": target.index,
            "query_frame_index": source.index,
            "camera_distance_m": camera_distance,
            "accepted": False,
        }
        visual = visual_registration(
            source.features,
            source.points,
            source.transform,
            target.features,
            target.points,
            target.transform,
            config.sift_ratio,
            config.visual_ransac_px,
            config.visual_3d_threshold,
            config.visual_ransac_iterations,
            config.loop_min_inliers,
            config.visual_min_coverage,
            rng,
        )
        visual.pop("_diagnostics", None)
        report["visual"] = visual
        if not visual.get("success"):
            report["reason"] = "visual_verification_failed"
            candidate_reports.append(report)
            continue

        source_world = transform_points(source.points, source.transform)
        target_world = transform_points(target.points, target.transform)
        correction, geometry = icp_registration(
            source_world,
            target_world,
            visual["transform"],
            config.voxel_size,
            config.max_registration_points,
        )
        corrected_source = transform_points(source_world, correction)
        quality = nearest_neighbor_quality(
            corrected_source,
            target_world,
            config.overlap_distance,
            config.max_registration_points,
        )
        correction_m, correction_deg = transform_difference(
            correction,
            np.eye(4),
            reference_point=source.transform[:3, 3],
        )
        geometry.update(
            quality=quality,
            correction_m=correction_m,
            correction_deg=correction_deg,
        )
        report["geometry"] = geometry
        reasons = []
        if not geometry.get("success"):
            reasons.append("geometry_failed")
        if quality["overlap"] < config.loop_min_overlap:
            reasons.append("low_overlap")
        if quality["rmse"] is None or quality["rmse"] > config.loop_max_rmse:
            reasons.append("high_rmse")
        if correction_m > 2.0 * config.max_residual_translation:
            reasons.append("excessive_translation")
        if correction_deg > 2.0 * config.max_residual_rotation_deg:
            reasons.append("excessive_rotation")
        if reasons:
            report["reason"] = ",".join(reasons)
            candidate_reports.append(report)
            continue

        corrected_source_pose = correction @ source.transform
        measurement = _relative_pose(target.transform, corrected_source_pose)
        weight = float(
            np.clip(
                0.5
                + quality["overlap"]
                + min(visual.get("inliers_3d", 0) / 50.0, 1.0),
                0.5,
                2.5,
            )
        )
        edges.append(
            {
                "source": first,
                "target": second,
                "kind": "loop",
                "measurement": measurement,
                "weight": weight,
            }
        )
        report.update(accepted=True, reason="ok", weight=weight)
        candidate_reports.append(report)
    return edges, candidate_reports


def optimize_pose_graph(
    keyframes: List[Keyframe], config, rng: np.random.Generator
) -> Tuple[List[np.ndarray], Dict]:
    """Optimize all camera poses while fixing the first pose as gauge origin."""
    initial = [np.asarray(frame.transform, dtype=np.float64).copy() for frame in keyframes]
    if len(initial) < 2:
        return initial, {
            "status": "insufficient_keyframes",
            "optimized": False,
            "candidate_loops": 0,
            "accepted_loops": 0,
            "edges": [],
        }
    edges, candidates = build_pose_graph(keyframes, config, rng)
    loop_edges = [edge for edge in edges if edge["kind"] == "loop"]
    before = _graph_metrics(initial, edges)
    if not loop_edges:
        return initial, {
            "status": "no_verified_loop",
            "optimized": False,
            "candidate_loops": len(candidates),
            "accepted_loops": 0,
            "before": before,
            "after": before,
            "candidates": candidates,
            "edges": edges,
        }

    fixed = initial[0]
    x0 = np.concatenate([_pack_pose(pose) for pose in initial[1:]])
    translation_sigma = max(float(config.pose_graph_translation_sigma), 1e-6)
    rotation_sigma = max(np.radians(config.pose_graph_rotation_sigma_deg), 1e-6)

    def unpack_all(values):
        return [fixed] + [
            _unpack_pose(values[offset : offset + 6])
            for offset in range(0, len(values), 6)
        ]

    def residuals(values):
        poses = unpack_all(values)
        output = []
        for edge in edges:
            translation, rotation = _edge_error(
                poses[edge["source"]],
                poses[edge["target"]],
                edge["measurement"],
            )
            weight = np.sqrt(float(edge["weight"]))
            output.extend(weight * translation / translation_sigma)
            output.extend(weight * rotation / rotation_sigma)
        return np.asarray(output, dtype=np.float64)

    solution = least_squares(
        residuals,
        x0,
        loss="soft_l1",
        f_scale=max(float(config.pose_graph_robust_scale), 1e-6),
        max_nfev=int(config.pose_graph_max_iterations),
    )
    optimized = unpack_all(solution.x)
    after = _graph_metrics(optimized, edges)
    translations = [
        float(np.linalg.norm(new[:3, 3] - old[:3, 3]))
        for old, new in zip(initial, optimized)
    ]
    rotations = [
        rotation_angle_deg(new[:3, :3] @ old[:3, :3].T)
        for old, new in zip(initial, optimized)
    ]
    report = {
        "status": "optimized" if solution.success else "optimizer_stopped",
        "optimized": bool(solution.success),
        "optimizer_message": str(solution.message),
        "function_evaluations": int(solution.nfev),
        "candidate_loops": len(candidates),
        "accepted_loops": len(loop_edges),
        "before": before,
        "after": after,
        "maximum_pose_translation_change_m": max(translations),
        "maximum_pose_rotation_change_deg": max(rotations),
        "candidates": candidates,
        "edges": edges,
        "initial_poses": initial,
        "optimized_poses": optimized,
        "edge_convention": (
            "measurement maps target-node local coordinates into source-node "
            "local coordinates: inverse(T_source) @ T_target"
        ),
    }
    return optimized, report
