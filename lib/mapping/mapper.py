"""Incremental pose refinement and confidence-weighted map fusion."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:  # MapIO will provide the user-facing dependency error first.
    o3d = None

from .config import MappingConfig
from .registration import (
    extract_sift_features,
    ground_plane_alignment,
    icp_registration,
    make_translation,
    nearest_neighbor_quality,
    rotation_angle_deg,
    transform_difference,
    transform_points,
    visual_registration,
)
from .types import FrameData, FrameResult, Keyframe, _jsonable
from .voxel_map import VoxelMap


class IncrementalMapper:
    def __init__(self, config: MappingConfig):
        self.config = config
        self.map = VoxelMap(
            config.voxel_size,
            config.max_voxel_weight,
            config.color_sigma,
            config.color_reject_distance,
        )
        self.ground_map = VoxelMap(
            config.ground_voxel_size,
            config.max_voxel_weight,
            config.color_sigma,
            config.color_reject_distance,
        )
        self.keyframes: List[Keyframe] = []
        self.results: List[FrameResult] = []
        self.rng = np.random.default_rng(config.random_seed)
        self.origin_utm: Optional[np.ndarray] = None

    def _prior_transform(self, frame: FrameData) -> np.ndarray:
        if self.config.position_mode == "relative":
            return make_translation(frame.relative_position)
        if self.keyframes:
            # With positions disabled, keep the latest registered camera origin
            # as the visual/geometric search seed.  SIFT must recover motion.
            return self.keyframes[-1].transform.copy()
        return np.eye(4, dtype=np.float64)

    def _features(self, frame: FrameData):
        if not self.config.use_visual:
            return None
        return extract_sift_features(
            frame.image,
            self.config.sift_features,
            self.config.sift_max_dimension,
        )

    def _candidate_keyframes(self, frame: FrameData) -> List[Keyframe]:
        count = max(1, int(self.config.max_keyframe_candidates))
        if self.config.position_mode == "relative":
            return sorted(
                self.keyframes,
                key=lambda keyframe: float(
                    np.linalg.norm(keyframe.relative_position - frame.relative_position)
                ),
            )[:count]
        return list(reversed(self.keyframes[-count:]))

    def _best_visual_registration(
        self,
        frame: FrameData,
        features,
        prior_transform: np.ndarray,
    ) -> Dict:
        if not self.config.use_visual:
            return {
                "success": False,
                "reason": "disabled",
                "transform": np.eye(4, dtype=np.float64),
                "candidates_tested": 0,
            }
        candidates = self._candidate_keyframes(frame)
        if not candidates:
            return {
                "success": False,
                "reason": "no_keyframe",
                "transform": np.eye(4, dtype=np.float64),
                "candidates_tested": 0,
            }

        reports = []
        for keyframe in candidates:
            report = visual_registration(
                features,
                frame.points,
                prior_transform,
                keyframe.features,
                keyframe.points,
                keyframe.transform,
                self.config.sift_ratio,
                self.config.visual_ransac_px,
                self.config.visual_3d_threshold,
                self.config.visual_ransac_iterations,
                self.config.visual_min_inliers,
                self.config.visual_min_coverage,
                self.rng,
            )
            report["keyframe_index"] = keyframe.index
            report["keyframe_name"] = keyframe.name
            reports.append(report)

        def score(report):
            rmse = report.get("rmse_3d")
            return (
                int(bool(report.get("success"))),
                int(report.get("inliers_3d", 0)),
                float(report.get("coverage", 0.0)),
                -float(rmse if rmse is not None else np.inf),
            )

        best = max(reports, key=score)
        best["candidates_tested"] = len(reports)
        best["candidate_summaries"] = [
            {
                "keyframe_index": item["keyframe_index"],
                "success": bool(item.get("success")),
                "matches": int(item.get("matches", 0)),
                "inliers_3d": int(item.get("inliers_3d", 0)),
                "reason": item.get("reason"),
            }
            for item in reports
        ]
        return best

    def _bounds(self, points: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return np.full(3, -np.inf), np.full(3, np.inf)
        # Ignore a tiny number of extreme monocular-depth outliers when
        # selecting a local submap; they remain available for later fusion.
        minimum = np.quantile(points, 0.002, axis=0) - margin
        maximum = np.quantile(points, 0.998, axis=0) + margin
        return minimum, maximum

    def _local_scene(self, placed_points: np.ndarray):
        minimum, maximum = self._bounds(placed_points, self.config.local_map_margin)
        return self.map.extract_bounds(minimum, maximum)

    def _local_ground(self, placed_ground: np.ndarray):
        minimum, maximum = self._bounds(placed_ground, self.config.local_map_margin)
        # The GEP constraint needs spatially corresponding ground but permits
        # more vertical variation than the scene crop.
        minimum[2] -= self.config.local_map_margin
        maximum[2] += self.config.local_map_margin
        return self.ground_map.extract_bounds(minimum, maximum)

    def _clean_points(self, frame: FrameData):
        points = np.asarray(frame.points, dtype=np.float64)
        valid = np.isfinite(points).all(axis=1)
        clean_points = points[valid]
        clean_colors = None
        if frame.colors is not None:
            colors = np.asarray(frame.colors, dtype=np.float64)
            if colors.shape == points.shape:
                clean_colors = colors[valid]
        return clean_points, clean_colors

    def _integrate(
        self,
        frame: FrameData,
        transform: np.ndarray,
        confidence: float,
    ):
        points, colors = self._clean_points(frame)
        mapped_points = transform_points(points, transform)
        self.map.integrate(mapped_points, colors, confidence, frame.index)
        if frame.ground_points is not None:
            ground = np.asarray(frame.ground_points, dtype=np.float64)
            ground = ground[np.isfinite(ground).all(axis=1)]
            if len(ground):
                self.ground_map.integrate(
                    transform_points(ground, transform),
                    None,
                    confidence,
                    frame.index,
                )

    def _add_keyframe(self, frame: FrameData, transform: np.ndarray, features):
        self.keyframes.append(
            Keyframe(
                index=frame.index,
                name=frame.name,
                position_utm=frame.position_utm.copy(),
                relative_position=frame.relative_position.copy(),
                points=np.asarray(frame.points, dtype=np.float64).copy(),
                transform=np.asarray(transform, dtype=np.float64).copy(),
                features=features,
            )
        )

    def _finish_result(
        self,
        frame: FrameData,
        result: FrameResult,
        features,
    ) -> FrameResult:
        if result.accepted:
            self._integrate(frame, result.transform, result.confidence)
            self._add_keyframe(frame, result.transform, features)
        result.map_voxels = len(self.map)
        self.results.append(result)
        if self.config.do_save:
            self._save_checkpoint(result)
        rmse_text = "n/a" if result.rmse_after is None else f"{result.rmse_after:.3f} m"
        print(
            f"[mapping] frame {result.index} {result.status}: "
            f"overlap={result.overlap_after:.3f}, rmse={rmse_text}, "
            f"map={result.map_voxels} voxels"
        )
        return result

    def process(self, frame: FrameData) -> FrameResult:
        if self.origin_utm is None:
            self.origin_utm = frame.position_utm - frame.relative_position
        clean_points, _ = self._clean_points(frame)
        if len(clean_points) < 20:
            raise ValueError(f"frame {frame.name} has fewer than 20 finite points")

        prior = self._prior_transform(frame)
        features = self._features(frame)
        identity = np.eye(4, dtype=np.float64)

        if len(self.map) == 0:
            result = FrameResult(
                index=frame.index,
                name=frame.name,
                accepted=True,
                status="initialized",
                transform=prior,
                prior_transform=prior,
                residual_transform=identity,
                confidence=1.0,
                overlap_after=1.0,
                message="First frame initialized the cumulative map.",
            )
            return self._finish_result(frame, result, features)

        source_prior = transform_points(clean_points, prior)
        all_map_points, _ = self.map.to_arrays()
        before_quality = nearest_neighbor_quality(
            source_prior,
            all_map_points,
            self.config.overlap_distance,
            self.config.max_registration_points,
        )

        visual = self._best_visual_registration(frame, features, prior)
        seed = visual["transform"] if visual.get("success") else identity
        source_seeded = transform_points(source_prior, seed)
        target_points, _ = self._local_scene(source_seeded)
        seed_quality = nearest_neighbor_quality(
            source_seeded,
            target_points,
            self.config.overlap_distance,
            self.config.max_registration_points,
        )

        if seed_quality["overlap"] < self.config.min_overlap:
            if self.config.position_mode == "relative" and not visual.get("success"):
                result = FrameResult(
                    index=frame.index,
                    name=frame.name,
                    accepted=True,
                    status="position_only_component",
                    transform=prior,
                    prior_transform=prior,
                    residual_transform=identity,
                    confidence=0.35,
                    overlap_before=before_quality["overlap"],
                    overlap_after=before_quality["overlap"],
                    rmse_after=before_quality["rmse"],
                    visual=visual,
                    message=(
                        "No reliable overlap was found; frame was added as a new "
                        "component using its relative position prior."
                    ),
                )
                return self._finish_result(frame, result, features)

            result = FrameResult(
                index=frame.index,
                name=frame.name,
                accepted=False,
                status="rejected_no_overlap",
                transform=seed @ prior,
                prior_transform=prior,
                residual_transform=seed,
                confidence=0.0,
                overlap_before=before_quality["overlap"],
                overlap_after=seed_quality["overlap"],
                rmse_after=seed_quality["rmse"],
                visual=visual,
                message="No reliable overlap or safe position-only placement was available.",
            )
            return self._finish_result(frame, result, features)

        ground_report: Dict = {"used": False, "reason": "disabled_or_unavailable"}
        registration_seed = seed
        if (
            self.config.use_ground
            and frame.ground_points is not None
            and len(self.ground_map) > 0
        ):
            ground = np.asarray(frame.ground_points, dtype=np.float64)
            ground = ground[np.isfinite(ground).all(axis=1)]
            ground_prior = transform_points(ground, prior)
            ground_seeded = transform_points(ground_prior, seed)
            target_ground, _ = self._local_ground(ground_seeded)
            registration_seed, ground_report = ground_plane_alignment(
                ground_prior,
                target_ground,
                seed,
                self.config.ground_min_points,
                self.config.ground_max_tilt_deg,
                self.config.ground_max_shift,
            )

        placed_for_crop = transform_points(source_prior, registration_seed)
        target_points, _ = self._local_scene(placed_for_crop)
        residual, geometry_report = icp_registration(
            source_prior,
            target_points,
            registration_seed,
            self.config.voxel_size,
            self.config.max_registration_points,
        )
        source_registered = transform_points(source_prior, residual)
        after_quality = nearest_neighbor_quality(
            source_registered,
            target_points,
            self.config.overlap_distance,
            self.config.max_registration_points,
        )

        prior_camera = np.asarray(prior[:3, 3], dtype=np.float64)
        residual_translation, _ = transform_difference(
            residual, identity, reference_point=prior_camera
        )
        residual_rotation = rotation_angle_deg(residual[:3, :3])
        geometry_report.update(
            residual_translation=residual_translation,
            residual_rotation_deg=residual_rotation,
            quality=after_quality,
        )
        last_level = geometry_report.get("levels", [])[-1:] or [{}]
        icp_fitness = float(last_level[0].get("fitness", 0.0))
        rmse = after_quality["rmse"]

        rejection_reasons = []
        if not geometry_report.get("success"):
            rejection_reasons.append(geometry_report.get("reason", "icp_failed"))
        if after_quality["overlap"] < self.config.min_overlap:
            rejection_reasons.append("low_overlap")
        if icp_fitness < self.config.min_icp_fitness:
            rejection_reasons.append("low_icp_fitness")
        if rmse is None or rmse > self.config.max_icp_rmse:
            rejection_reasons.append("high_rmse")
        if residual_translation > self.config.max_residual_translation:
            rejection_reasons.append("translation_exceeds_prior_limit")
        if residual_rotation > self.config.max_residual_rotation_deg:
            rejection_reasons.append("rotation_exceeds_imu_limit")

        if visual.get("success"):
            disagreement_m, disagreement_deg = transform_difference(
                residual, seed, reference_point=prior_camera
            )
            visual["icp_disagreement_m"] = disagreement_m
            visual["icp_disagreement_deg"] = disagreement_deg
            if disagreement_m > self.config.max_visual_icp_disagreement_m:
                rejection_reasons.append("visual_geometry_translation_disagree")
            if disagreement_deg > self.config.max_visual_icp_disagreement_deg:
                rejection_reasons.append("visual_geometry_rotation_disagree")

        accepted = not rejection_reasons
        if accepted:
            geometry_strength = min(
                1.0,
                after_quality["overlap"] / max(0.25, self.config.min_overlap),
            )
            geometry_strength *= np.exp(
                -float(rmse) / max(2.0 * self.config.voxel_size, 1e-9)
            )
            visual_strength = 0.0
            if visual.get("success"):
                visual_strength = min(1.0, visual.get("inliers_3d", 0) / 50.0)
            confidence = float(
                np.clip(0.25 + 0.60 * geometry_strength + 0.15 * visual_strength, 0.25, 1.0)
            )
            status = "registered_visual_geometry" if visual.get("success") else "registered_geometry"
            message = "Registration passed overlap, residual-pose, and RMSE checks."
        else:
            confidence = 0.0
            status = "rejected_registration"
            message = "Registration rejected: " + ", ".join(rejection_reasons)

        final_transform = residual @ prior
        result = FrameResult(
            index=frame.index,
            name=frame.name,
            accepted=accepted,
            status=status,
            transform=final_transform,
            prior_transform=prior,
            residual_transform=residual,
            confidence=confidence,
            overlap_before=before_quality["overlap"],
            overlap_after=after_quality["overlap"],
            rmse_after=rmse,
            visual=visual,
            ground=ground_report,
            geometry=geometry_report,
            message=message,
        )
        return self._finish_result(frame, result, features)

    def _write_cloud_atomic(self, path: Path, points: np.ndarray, colors: np.ndarray):
        if o3d is None:
            raise RuntimeError("Open3D is required to save mapping output")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        if colors is not None and len(colors) == len(points):
            cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        success = o3d.io.write_point_cloud(
            str(temporary), cloud, write_ascii=False, compressed=True
        )
        if not success:
            raise IOError(f"failed to write point cloud: {temporary}")
        os.replace(temporary, path)

    @staticmethod
    def _write_text_atomic(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)

    def _save_state_atomic(self):
        state = {}
        state.update(self.map.state_arrays("map"))
        state.update(self.ground_map.state_arrays("ground"))
        state["origin_utm"] = np.asarray(self.origin_utm, dtype=np.float64)
        state["frame_indices"] = np.asarray([item.index for item in self.results], dtype=np.int32)
        state["frame_names"] = np.asarray([item.name for item in self.results], dtype=str)
        state["accepted"] = np.asarray([item.accepted for item in self.results], dtype=bool)
        state["transforms"] = np.asarray([item.transform for item in self.results], dtype=np.float64)
        path = self.config.output_dir / "state.npz"
        temporary = path.with_name("state.tmp.npz")
        np.savez_compressed(temporary, **state)
        os.replace(temporary, path)

    def _save_checkpoint(self, result: FrameResult):
        points, colors = self.map.to_arrays()
        self._write_cloud_atomic(self.config.output_dir / "map.pcd", points, colors)
        step_name = f"{result.index:06d}_{result.name}.pcd"
        self._write_cloud_atomic(self.config.steps_dir / step_name, points, colors)
        self._write_text_atomic(
            self.config.reports_dir / f"{result.index:06d}_{result.name}.json",
            json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        )
        trajectory = "".join(json.dumps(item.as_dict(), sort_keys=True) + "\n" for item in self.results)
        self._write_text_atomic(self.config.output_dir / "trajectory.jsonl", trajectory)
        self._save_state_atomic()
        coordinate_frame = (
            "fused world axes, meters, relative to origin_utm"
            if self.config.position_mode == "relative"
            else "fused world axes, meters, relative to first processed frame"
        )
        origin = {
            "coordinate_frame": coordinate_frame,
            "origin_utm": _jsonable(self.origin_utm),
            "position_mode": self.config.position_mode,
            "utm_zone": self.config.utm_zone,
        }
        self._write_text_atomic(
            self.config.output_dir / "origin.json",
            json.dumps(origin, indent=2, sort_keys=True) + "\n",
        )

    def finalize(self) -> Dict:
        summary = {
            "frames_processed": len(self.results),
            "frames_accepted": sum(item.accepted for item in self.results),
            "frames_rejected": sum(not item.accepted for item in self.results),
            "map_voxels": len(self.map),
            "voxel_size_m": self.config.voxel_size,
            "position_mode": self.config.position_mode,
            "output_dir": str(self.config.output_dir),
        }
        if self.config.do_save:
            self._write_text_atomic(
                self.config.output_dir / "summary.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
            )
        print(
            f"[mapping] complete: {summary['frames_accepted']}/{summary['frames_processed']} "
            f"frames accepted, {summary['map_voxels']} voxels"
        )
        return summary
