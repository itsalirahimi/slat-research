"""Incremental pose refinement and confidence-weighted map fusion."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import open3d as o3d
except ImportError:  # MapIO will provide the user-facing dependency error first.
    o3d = None

from .config import MappingConfig
from .diagnostics import (
    geometry_error_cloud,
    trajectory_image,
    visual_diagnostic_images,
)
from .loop_closure import optimize_pose_graph
from .occupancy3d import OccupancyGrid3D
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
from .products import (
    RgbMap2D,
    SurfaceMesh,
    build_height_field_mesh,
    build_rgb_map_2d,
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
        self.latest_mesh: Optional[SurfaceMesh] = None
        self.latest_rgb_map: Optional[RgbMap2D] = None
        self.occupancy = self._make_occupancy_grid() if config.save_occupancy else None
        self._visual_diagnostics: Dict[int, Dict] = {}
        self._geometry_diagnostics: Dict[int, Tuple[np.ndarray, np.ndarray, Dict]] = {}
        self._occupancy_updates: Dict[int, Dict] = {}
        self.loop_closure_report: Optional[Dict] = None
        self.timing_records: List[Dict] = []
        self.final_timing: Dict[str, float] = {}

    def _make_occupancy_grid(self) -> OccupancyGrid3D:
        return OccupancyGrid3D(
            self.config.occupancy_voxel_size,
            self.config.occupancy_free_log_odds,
            self.config.occupancy_occupied_log_odds,
            self.config.occupancy_free_threshold,
            self.config.occupancy_occupied_threshold,
            self.config.occupancy_clamp_min,
            self.config.occupancy_clamp_max,
            self.config.occupancy_max_rays_per_frame,
            self.config.occupancy_max_ray_length,
            self.config.occupancy_ray_step_factor,
        )

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
        features = extract_sift_features(
            frame.image,
            self.config.sift_features,
            self.config.sift_max_dimension,
        )
        if (
            features is not None
            and len(frame.points) == int(np.prod(features.image_shape))
            and len(features.points_xy)
        ):
            height, width = features.image_shape
            u = np.rint(features.points_xy[:, 0]).astype(np.int64)
            v = np.rint(features.points_xy[:, 1]).astype(np.int64)
            valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            lifted = np.full((len(features.points_xy), 3), np.nan, dtype=np.float64)
            lifted[valid] = np.asarray(frame.points, dtype=np.float64)[
                v[valid] * width + u[valid]
            ]
            features.points_3d = lifted
        return features

    @staticmethod
    def _sample_keyframe_cloud(
        points: np.ndarray,
        colors: Optional[np.ndarray],
        voxel_size: float,
        max_points: int,
    ):
        """Keep a bounded local cloud; SIFT correspondences are stored separately."""
        original = np.asarray(points, dtype=np.float64)
        valid = np.isfinite(original).all(axis=1)
        clean = original[valid]
        clean_colors = None
        if colors is not None:
            candidate = np.asarray(colors, dtype=np.float64)
            if candidate.shape == original.shape:
                clean_colors = candidate[valid]
        if not len(clean):
            return clean, clean_colors
        keys = np.floor(clean / max(float(voxel_size), 1e-6)).astype(np.int64)
        _, selected = np.unique(keys, axis=0, return_index=True)
        selected.sort()
        if len(selected) > max_points:
            selected = selected[
                np.linspace(0, len(selected) - 1, max_points, dtype=np.int64)
            ]
        return clean[selected], None if clean_colors is None else clean_colors[selected]

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
        diagnostic = best.pop("_diagnostics", None)
        if (
            diagnostic is not None
            and self.config.do_save
            and self.config.save_diagnostics
        ):
            target = next(
                item for item in candidates if item.index == best["keyframe_index"]
            )
            target_image = target.image
            if target_image is None and target.image_path and cv2 is not None:
                target_image = cv2.imread(target.image_path, cv2.IMREAD_COLOR)
            diagnostic["source_image"] = frame.image
            diagnostic["target_image"] = target_image
            self._visual_diagnostics[frame.index] = diagnostic
        for report in reports:
            report.pop("_diagnostics", None)
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
        if self.occupancy is not None:
            camera_origin = np.asarray(transform[:3, 3], dtype=np.float64)
            self._occupancy_updates[frame.index] = self.occupancy.integrate(
                camera_origin,
                mapped_points,
                confidence,
                frame.index,
            )
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

    def _add_keyframe(
        self, frame: FrameData, transform: np.ndarray, features, confidence: float
    ):
        sampled_points, sampled_colors = self._sample_keyframe_cloud(
            frame.points,
            frame.colors,
            0.5 * self.config.voxel_size,
            2 * self.config.max_registration_points,
        )
        sampled_ground = None
        if frame.ground_points is not None:
            sampled_ground, _ = self._sample_keyframe_cloud(
                frame.ground_points,
                None,
                self.config.ground_voxel_size,
                self.config.max_registration_points,
            )
        retained_image = None
        if frame.image_path is None and frame.image is not None and frame.image.size <= 12_000_000:
            retained_image = np.asarray(frame.image, dtype=np.uint8).copy()
        self.keyframes.append(
            Keyframe(
                index=frame.index,
                name=frame.name,
                position_utm=frame.position_utm.copy(),
                relative_position=frame.relative_position.copy(),
                points=sampled_points,
                colors=sampled_colors,
                image=retained_image,
                image_path=frame.image_path,
                ground_points=sampled_ground,
                transform=np.asarray(transform, dtype=np.float64).copy(),
                features=features,
                confidence=float(confidence),
            )
        )

    def _finish_result(
        self,
        frame: FrameData,
        result: FrameResult,
        features,
        mapping_started: float,
        input_load_seconds: float,
    ) -> FrameResult:
        if result.accepted:
            self._integrate(frame, result.transform, result.confidence)
            self._add_keyframe(frame, result.transform, features, result.confidence)
        result.map_voxels = len(self.map)
        self.results.append(result)

        mapping_seconds = time.perf_counter() - mapping_started
        product_seconds = 0.0
        saving_seconds = 0.0
        report_seconds = 0.0
        if self.config.do_save:
            product_seconds, saving_seconds = self._save_checkpoint(result)
            report_started = time.perf_counter()
            if self.config.save_diagnostics:
                self._save_registration_diagnostics(result)
            # The timing JSONL written below is authoritative.  This first
            # report write is included in report_seconds; it is refreshed once
            # after measuring so the human-readable frame report also contains
            # the measured value.
            report_path = self.config.reports_dir / (
                f"{result.index:06d}_{result.name}.json"
            )
            self._write_text_atomic(
                report_path,
                json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
            )
            report_seconds = time.perf_counter() - report_started

        result.timing = {
            "input_load_seconds": float(input_load_seconds),
            "mapping_seconds": float(mapping_seconds),
            "derived_products_seconds": float(product_seconds),
            "checkpoint_saving_seconds": float(saving_seconds),
            "report_generation_seconds": float(report_seconds),
        }
        result.timing["frame_work_total_seconds"] = float(
            sum(result.timing.values())
        )
        timing_record = {
            "index": result.index,
            "name": result.name,
            "status": result.status,
            **result.timing,
        }
        self.timing_records.append(timing_record)
        if self.config.do_save:
            # Refresh the report with finalized timing.  Timing metadata/console
            # logging itself is deliberately excluded from measured work.
            self._write_text_atomic(
                self.config.reports_dir / f"{result.index:06d}_{result.name}.json",
                json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
            )
            timings = "".join(
                json.dumps(item, sort_keys=True) + "\n"
                for item in self.timing_records
            )
            self._write_text_atomic(self.config.output_dir / "timings.jsonl", timings)

        rmse_text = "n/a" if result.rmse_after is None else f"{result.rmse_after:.3f} m"
        print(
            f"[mapping] frame {result.index} {result.status}: "
            f"overlap={result.overlap_after:.3f}, rmse={rmse_text}, "
            f"map={result.map_voxels} voxels"
        )
        print(
            f"[timing] frame {result.index:06d}: "
            f"load={input_load_seconds:.6f}s "
            f"mapping={mapping_seconds:.6f}s "
            f"products={product_seconds:.6f}s "
            f"saving={saving_seconds:.6f}s "
            f"reports={report_seconds:.6f}s "
            f"total={result.timing['frame_work_total_seconds']:.6f}s"
        )
        return result

    def process(
        self, frame: FrameData, input_load_seconds: float = 0.0
    ) -> FrameResult:
        mapping_started = time.perf_counter()
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
            return self._finish_result(
                frame, result, features, mapping_started, input_load_seconds
            )

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
                return self._finish_result(
                    frame, result, features, mapping_started, input_load_seconds
                )

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
            return self._finish_result(
                frame, result, features, mapping_started, input_load_seconds
            )

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
        if self.config.do_save and self.config.save_diagnostics:
            # Retain the two bounded registration clouds until checkpointing so
            # diagnostic construction is timed as report generation, not as
            # registration/mapping work.
            self._geometry_diagnostics[frame.index] = (
                source_registered,
                target_points,
                {},
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
        return self._finish_result(
            frame, result, features, mapping_started, input_load_seconds
        )

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

    def _write_mesh_atomic(self, path: Path, mesh_data: SurfaceMesh):
        if o3d is None:
            raise RuntimeError("Open3D is required to save mesh output")
        if mesh_data.is_empty:
            print(f"[mapping][warning] no safe mesh triangles for {path.name}; skipping")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(mesh_data.vertices)
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_data.colors)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_data.triangles)
        mesh.compute_vertex_normals()
        success = o3d.io.write_triangle_mesh(
            str(temporary),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
        if not success:
            raise IOError(f"failed to write triangle mesh: {temporary}")
        os.replace(temporary, path)

    @staticmethod
    def _write_text_atomic(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)

    @staticmethod
    def _write_bytes_atomic(path: Path, content: bytes):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(content)
        os.replace(temporary, path)

    @staticmethod
    def _write_image_atomic(path: Path, image: np.ndarray):
        if cv2 is None:
            raise RuntimeError("OpenCV is required to save mapping image products")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        if not cv2.imwrite(str(temporary), np.asarray(image)):
            raise IOError(f"failed to write image: {temporary}")
        os.replace(temporary, path)

    def _save_occupancy_atomic(self, base_path: Path, grid: OccupancyGrid3D):
        npz_path = base_path.with_suffix(".npz")
        temporary = npz_path.with_name(npz_path.stem + ".tmp.npz")
        state = grid.state_arrays("occupancy3d")
        state["values"] = np.asarray([-1, 0, 50, 100], dtype=np.int16)
        state["labels"] = np.asarray(
            ["unknown_implicit", "free", "known_uncertain", "occupied"]
        )
        np.savez_compressed(temporary, **state)
        os.replace(temporary, npz_path)
        visible_points, visible_colors, _ = grid.visualization_arrays(
            self.config.occupancy_visual_max_points
        )
        if len(visible_points):
            self._write_cloud_atomic(
                base_path.with_suffix(".pcd"), visible_points, visible_colors
            )
        metadata = grid.summary()
        metadata.update(
            values={
                "-1": "unknown (implicit/not stored)",
                "0": "free",
                "50": "known but uncertain",
                "100": "occupied",
            },
            visualization={
                "free": "transparent/omitted",
                "known_uncertain": "light orange",
                "occupied": "light red",
            },
            pcd_representation=(
                "Colored centers of visible voxels. Use launch/view3d.py to "
                "render each center as a cube with this voxel size."
            ),
        )
        self._write_text_atomic(
            base_path.with_suffix(".json"),
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )

    def _build_derived_products(self, points: np.ndarray, colors: np.ndarray):
        if self.config.save_mesh:
            self.latest_mesh = build_height_field_mesh(
                points,
                colors,
                self.config.mesh_grid_size,
                self.config.mesh_max_height_jump,
                self.config.mesh_max_edge_factor,
            )
        if self.config.save_map_2d:
            self.latest_rgb_map = build_rgb_map_2d(
                points,
                colors,
                self.config.map_2d_resolution,
                self.config.map_2d_padding,
                self.config.map_2d_max_pixels,
            )

    @staticmethod
    def _trajectory_arrays(results: List[FrameResult]):
        if not results:
            return np.empty((0, 3)), np.empty(0, dtype=bool)
        points = np.asarray([item.transform[:3, 3] for item in results], dtype=np.float64)
        accepted = np.asarray([item.accepted for item in results], dtype=bool)
        return points, accepted

    def _write_trajectory_ply_atomic(
        self, path: Path, points: np.ndarray, accepted: np.ndarray
    ):
        points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        accepted = np.asarray(accepted, dtype=bool).reshape(-1)
        edges = [
            (index - 1, index)
            for index in range(1, len(points))
            if accepted[index - 1] and accepted[index]
        ]
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property double x",
            "property double y",
            "property double z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element edge {len(edges)}",
            "property int vertex1",
            "property int vertex2",
            "end_header",
        ]
        for point, keep in zip(points, accepted):
            color = (70, 240, 70) if keep else (245, 70, 70)
            lines.append(
                f"{point[0]:.12g} {point[1]:.12g} {point[2]:.12g} "
                f"{color[0]} {color[1]} {color[2]}"
            )
        lines.extend(f"{first} {second}" for first, second in edges)
        self._write_text_atomic(path, "\n".join(lines) + "\n")

    def _save_rgb_map_atomic(
        self,
        directory: Path,
        prefix: str,
        rgb_map: RgbMap2D,
        trajectory_points: np.ndarray,
        accepted: np.ndarray,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self._write_image_atomic(
            directory / f"{prefix}_rgb.png",
            cv2.cvtColor(rgb_map.image_rgb, cv2.COLOR_RGB2BGR),
        )
        self._write_image_atomic(directory / f"{prefix}_mask.png", rgb_map.mask)
        self._write_image_atomic(
            directory / f"{prefix}_trajectory.png",
            trajectory_image(rgb_map, trajectory_points, accepted),
        )
        data_path = directory / f"{prefix}_data.npz"
        temporary = data_path.with_name(data_path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary,
            image_rgb=rgb_map.image_rgb,
            mask=rgb_map.mask,
            elevation=rgb_map.elevation,
            origin_xy=rgb_map.origin_xy,
            resolution=np.asarray(rgb_map.resolution, dtype=np.float64),
        )
        os.replace(temporary, data_path)
        self._write_text_atomic(
            directory / f"{prefix}_metadata.json",
            json.dumps(rgb_map.summary(), indent=2, sort_keys=True) + "\n",
        )

    def _save_registration_diagnostics(self, result: FrameResult):
        diagnostic = self._visual_diagnostics.pop(result.index, None)
        if diagnostic is not None:
            images = visual_diagnostic_images(
                diagnostic.pop("source_image", None),
                diagnostic.pop("target_image", None),
                diagnostic,
                self.config.diagnostic_max_matches,
                self.config.diagnostic_max_image_dimension,
            )
            files = {}
            stem = f"{result.index:06d}_{result.name}"
            for kind, image in images.items():
                path = self.config.visual_reports_dir / f"{stem}_{kind}.png"
                self._write_image_atomic(path, image)
                files[kind] = str(path.relative_to(self.config.output_dir))
            if files:
                result.visual["diagnostic_files"] = files

        geometry = self._geometry_diagnostics.pop(result.index, None)
        geometry_file = None
        if geometry is not None and len(geometry[0]):
            diagnostic_points, diagnostic_colors, diagnostic_report = (
                geometry_error_cloud(
                    geometry[0],
                    geometry[1],
                    self.config.overlap_distance,
                    self.config.diagnostic_max_geometry_points,
                )
            )
            result.geometry["diagnostic"] = diagnostic_report
            geometry_file = self.config.geometry_reports_dir / (
                f"{result.index:06d}_{result.name}_alignment.ply"
            )
            self._write_cloud_atomic(
                geometry_file, diagnostic_points, diagnostic_colors
            )
            result.geometry.setdefault("diagnostic_files", {})["alignment_cloud"] = str(
                geometry_file.relative_to(self.config.output_dir)
            )
        geometry_payload = {
            "index": result.index,
            "name": result.name,
            "accepted": result.accepted,
            "status": result.status,
            "overlap_before": result.overlap_before,
            "overlap_after": result.overlap_after,
            "rmse_after": result.rmse_after,
            "matching": result.geometry or {
                "success": False,
                "reason": "not_run_for_this_frame_status",
            },
        }
        self._write_text_atomic(
            self.config.geometry_reports_dir
            / f"{result.index:06d}_{result.name}_geometry.json",
            json.dumps(_jsonable(geometry_payload), indent=2, sort_keys=True) + "\n",
        )

    def _save_state_atomic(self):
        state = {}
        state.update(self.map.state_arrays("map"))
        state.update(self.ground_map.state_arrays("ground"))
        if self.occupancy is not None:
            state.update(self.occupancy.state_arrays("occupancy3d"))
        state["origin_utm"] = np.asarray(self.origin_utm, dtype=np.float64)
        state["frame_indices"] = np.asarray([item.index for item in self.results], dtype=np.int32)
        state["frame_names"] = np.asarray([item.name for item in self.results], dtype=str)
        state["accepted"] = np.asarray([item.accepted for item in self.results], dtype=bool)
        state["transforms"] = np.asarray([item.transform for item in self.results], dtype=np.float64)
        path = self.config.output_dir / "state.npz"
        temporary = path.with_name("state.tmp.npz")
        np.savez_compressed(temporary, **state)
        os.replace(temporary, path)

    def _save_checkpoint(self, result: FrameResult) -> Tuple[float, float]:
        """Save the required cumulative PCD and optional heavy step products.

        The final map and all latest derived products are written by
        :meth:`finalize`.  This avoids writing the same full geometry twice per
        frame and avoids rebuilding products which are normally only consumed
        after the sequence completes.
        """
        saving_started = time.perf_counter()
        points, colors = self.map.to_arrays()
        step_stem = f"{result.index:06d}_{result.name}"
        self._write_cloud_atomic(
            self.config.steps_dir / f"{step_stem}.pcd", points, colors
        )
        saving_seconds = time.perf_counter() - saving_started
        product_seconds = 0.0

        if self.occupancy is not None:
            result.occupancy = self.occupancy.summary()
            result.occupancy["frame_update"] = self._occupancy_updates.get(
                result.index, {"rays": 0, "free_updates": 0, "occupied_updates": 0}
            )

        if self.config.save_derived_steps:
            products_started = time.perf_counter()
            self._build_derived_products(points, colors)
            product_seconds = time.perf_counter() - products_started
            trajectory_points, trajectory_accepted = self._trajectory_arrays(
                self.results
            )
            step_saving_started = time.perf_counter()
            if self.latest_mesh is not None:
                result.mesh = self.latest_mesh.summary()
                self._write_mesh_atomic(
                    self.config.mesh_steps_dir / f"{step_stem}.ply",
                    self.latest_mesh,
                )
            if self.occupancy is not None:
                self._save_occupancy_atomic(
                    self.config.occupancy_steps_dir / step_stem,
                    self.occupancy,
                )
            if self.latest_rgb_map is not None:
                result.map_2d = self.latest_rgb_map.summary()
                self._save_rgb_map_atomic(
                    self.config.map_2d_steps_dir,
                    step_stem,
                    self.latest_rgb_map,
                    trajectory_points,
                    trajectory_accepted,
                )
            saving_seconds += time.perf_counter() - step_saving_started
        return product_seconds, saving_seconds

    def _origin_payload(self) -> Dict:
        coordinate_frame = (
            "fused world axes, meters, relative to origin_utm"
            if self.config.position_mode == "relative"
            else "fused world axes, meters, relative to first processed frame"
        )
        return {
            "coordinate_frame": coordinate_frame,
            "origin_utm": _jsonable(self.origin_utm),
            "position_mode": self.config.position_mode,
            "utm_zone": self.config.utm_zone,
        }

    def _save_final_incremental_outputs(self) -> Tuple[float, float]:
        """Build latest derived products once and save the final checkpoint."""
        products_started = time.perf_counter()
        points, colors = self.map.to_arrays()
        self._build_derived_products(points, colors)
        trajectory_points, trajectory_accepted = self._trajectory_arrays(self.results)
        product_seconds = time.perf_counter() - products_started

        if self.results:
            if self.latest_mesh is not None:
                self.results[-1].mesh = self.latest_mesh.summary()
            if self.occupancy is not None:
                self.results[-1].occupancy = self.occupancy.summary()
                self.results[-1].occupancy["frame_update"] = (
                    self._occupancy_updates.get(
                        self.results[-1].index,
                        {"rays": 0, "free_updates": 0, "occupied_updates": 0},
                    )
                )
            if self.latest_rgb_map is not None:
                self.results[-1].map_2d = self.latest_rgb_map.summary()

        saving_started = time.perf_counter()
        self._write_cloud_atomic(self.config.output_dir / "map.pcd", points, colors)
        if self.latest_mesh is not None:
            self._write_mesh_atomic(
                self.config.output_dir / "map_mesh.ply", self.latest_mesh
            )
        if self.occupancy is not None:
            self._save_occupancy_atomic(
                self.config.occupancy_dir / "occupancy_grid_3d", self.occupancy
            )
        self._write_trajectory_ply_atomic(
            self.config.output_dir / "trajectory.ply",
            trajectory_points,
            trajectory_accepted,
        )
        if self.latest_rgb_map is not None:
            self._save_rgb_map_atomic(
                self.config.map_2d_dir,
                "map",
                self.latest_rgb_map,
                trajectory_points,
                trajectory_accepted,
            )
        trajectory = "".join(
            json.dumps(item.as_dict(), sort_keys=True) + "\n"
            for item in self.results
        )
        self._write_text_atomic(self.config.output_dir / "trajectory.jsonl", trajectory)
        self._save_state_atomic()
        self._write_text_atomic(
            self.config.output_dir / "origin.json",
            json.dumps(self._origin_payload(), indent=2, sort_keys=True) + "\n",
        )
        return product_seconds, time.perf_counter() - saving_started

    def _rebuild_with_poses(self, poses: List[np.ndarray]):
        optimized_map = VoxelMap(
            self.config.voxel_size,
            self.config.max_voxel_weight,
            self.config.color_sigma,
            self.config.color_reject_distance,
        )
        optimized_ground = VoxelMap(
            self.config.ground_voxel_size,
            self.config.max_voxel_weight,
            self.config.color_sigma,
            self.config.color_reject_distance,
        )
        optimized_occupancy = (
            self._make_occupancy_grid() if self.config.save_occupancy else None
        )
        for keyframe, pose in zip(self.keyframes, poses):
            points = np.asarray(keyframe.points, dtype=np.float64)
            mapped = transform_points(points, pose)
            optimized_map.integrate(
                mapped,
                keyframe.colors,
                keyframe.confidence,
                keyframe.index,
            )
            if optimized_occupancy is not None:
                optimized_occupancy.integrate(
                    pose[:3, 3], mapped, keyframe.confidence, keyframe.index
                )
            if keyframe.ground_points is not None:
                optimized_ground.integrate(
                    transform_points(keyframe.ground_points, pose),
                    None,
                    keyframe.confidence,
                    keyframe.index,
                )
        return optimized_map, optimized_ground, optimized_occupancy

    def _run_loop_closure(self) -> Dict:
        poses, graph_report = optimize_pose_graph(
            self.keyframes, self.config, self.rng
        )
        optimized_map, _, optimized_occupancy = self._rebuild_with_poses(poses)
        points, colors = optimized_map.to_arrays()
        mesh = None
        if self.config.save_mesh:
            mesh = build_height_field_mesh(
                points,
                colors,
                self.config.mesh_grid_size,
                self.config.mesh_max_height_jump,
                self.config.mesh_max_edge_factor,
            )
        rgb_map = None
        if self.config.save_map_2d:
            rgb_map = build_rgb_map_2d(
                points,
                colors,
                self.config.map_2d_resolution,
                self.config.map_2d_padding,
                self.config.map_2d_max_pixels,
            )

        before = graph_report.get("before", {})
        after = graph_report.get("after", {})
        before_rmse = before.get("translation_rmse_m")
        after_rmse = after.get("translation_rmse_m")
        improved = bool(
            before_rmse is not None
            and after_rmse is not None
            and after_rmse < before_rmse - 1e-9
        )
        comparison = {
            "status": graph_report.get("status"),
            "optimized": bool(graph_report.get("optimized", False)),
            "candidate_loops": int(graph_report.get("candidate_loops", 0)),
            "accepted_loops": int(graph_report.get("accepted_loops", 0)),
            "incremental_map_voxels": len(self.map),
            "optimized_map_voxels": len(optimized_map),
            "loop_residual_before": before,
            "loop_residual_after": after,
            "loop_residual_improved": improved,
            "maximum_pose_translation_change_m": graph_report.get(
                "maximum_pose_translation_change_m", 0.0
            ),
            "maximum_pose_rotation_change_deg": graph_report.get(
                "maximum_pose_rotation_change_deg", 0.0
            ),
            "accuracy_note": (
                "Lower verified-loop residual means better global internal consistency; "
                "it does not prove lower absolute error without ground truth."
            ),
        }
        self.loop_closure_report = comparison

        if self.config.do_save:
            output = self.config.optimized_dir
            output.mkdir(parents=True, exist_ok=True)
            self._write_cloud_atomic(output / "map.pcd", points, colors)
            if mesh is not None:
                self._write_mesh_atomic(output / "map_mesh.ply", mesh)
                comparison["mesh"] = mesh.summary()
            if optimized_occupancy is not None:
                self._save_occupancy_atomic(
                    output / "occupancy_grid_3d", optimized_occupancy
                )
                comparison["occupancy"] = optimized_occupancy.summary()
            pose_points = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
            accepted = np.ones(len(pose_points), dtype=bool)
            self._write_trajectory_ply_atomic(
                output / "trajectory.ply", pose_points, accepted
            )
            if rgb_map is not None:
                self._save_rgb_map_atomic(
                    output, "map", rgb_map, pose_points, accepted
                )
                comparison["map_2d"] = rgb_map.summary()
            np.savez_compressed(
                output / "poses.npz",
                frame_indices=np.asarray(
                    [frame.index for frame in self.keyframes], dtype=np.int32
                ),
                initial_poses=np.asarray(
                    [frame.transform for frame in self.keyframes], dtype=np.float64
                ),
                optimized_poses=np.asarray(poses, dtype=np.float64),
            )
            self._write_text_atomic(
                output / "pose_graph.json",
                json.dumps(_jsonable(graph_report), indent=2, sort_keys=True) + "\n",
            )
            self._write_text_atomic(
                output / "comparison.json",
                json.dumps(_jsonable(comparison), indent=2, sort_keys=True) + "\n",
            )
        return comparison

    @staticmethod
    def _aggregate_timings(records: List[Dict]) -> Dict:
        keys = (
            "input_load_seconds",
            "mapping_seconds",
            "derived_products_seconds",
            "checkpoint_saving_seconds",
            "report_generation_seconds",
            "frame_work_total_seconds",
        )
        if not records:
            return {
                "frame_count": 0,
                "totals_seconds": {key: 0.0 for key in keys},
                "means_seconds": {key: 0.0 for key in keys},
                "maxima_seconds": {key: 0.0 for key in keys},
            }
        return {
            "frame_count": len(records),
            "totals_seconds": {
                key: float(sum(float(item[key]) for item in records))
                for key in keys
            },
            "means_seconds": {
                key: float(np.mean([float(item[key]) for item in records]))
                for key in keys
            },
            "maxima_seconds": {
                key: float(max(float(item[key]) for item in records))
                for key in keys
            },
        }

    def finalize(self) -> Dict:
        final_products_seconds = 0.0
        final_saving_seconds = 0.0
        if self.config.do_save:
            final_products_seconds, final_saving_seconds = (
                self._save_final_incremental_outputs()
            )

        loop_started = time.perf_counter()
        loop_summary = self._run_loop_closure() if self.config.run_loop_closure else None
        loop_seconds = (
            time.perf_counter() - loop_started
            if self.config.run_loop_closure
            else 0.0
        )

        report_started = time.perf_counter()
        # Refresh reports after final product summaries and timing fields have
        # been populated.  Diagnostic rendering was already measured per frame.
        if self.config.do_save:
            for result in self.results:
                self._write_text_atomic(
                    self.config.reports_dir
                    / f"{result.index:06d}_{result.name}.json",
                    json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
                )
        final_report_seconds = time.perf_counter() - report_started

        frame_timing = self._aggregate_timings(self.timing_records)
        self.final_timing = {
            "derived_products_seconds": float(final_products_seconds),
            "saving_seconds": float(final_saving_seconds),
            "report_generation_seconds": float(final_report_seconds),
            "loop_closure_seconds": float(loop_seconds),
        }
        self.final_timing["finalization_work_total_seconds"] = float(
            sum(self.final_timing.values())
        )
        overall_seconds = (
            frame_timing["totals_seconds"]["frame_work_total_seconds"]
            + self.final_timing["finalization_work_total_seconds"]
        )
        timing_summary = {
            "clock": "time.perf_counter monotonic wall-clock",
            "precision": "seconds, stored as floating point and printed to 6 decimals",
            "viewer_wait": "excluded",
            "timing_log_and_console_overhead": "excluded",
            "phase_definitions": {
                "input_load": "Read PCD, RGB, ground cloud, and metadata for one frame.",
                "mapping": (
                    "Feature extraction/matching, geometric registration, occupancy "
                    "ray integration, voxel fusion, and keyframe update."
                ),
                "derived_products": "Build mesh and orthographic RGB arrays.",
                "saving": "Serialize non-report mapping products to storage.",
                "report_generation": (
                    "Create/save match overlays, geometry diagnostics, and frame reports."
                ),
                "loop_closure": "Verify loops, optimize the pose graph, and save optimized outputs.",
            },
            "frames": frame_timing,
            "finalization": self.final_timing,
            "overall_work_total_seconds": float(overall_seconds),
        }
        summary = {
            "frames_processed": len(self.results),
            "frames_accepted": sum(item.accepted for item in self.results),
            "frames_rejected": sum(not item.accepted for item in self.results),
            "map_voxels": len(self.map),
            "voxel_size_m": self.config.voxel_size,
            "position_mode": self.config.position_mode,
            "output_dir": str(self.config.output_dir),
            "mesh": self.latest_mesh.summary() if self.latest_mesh is not None else None,
            "occupancy": (
                self.occupancy.summary()
                if self.occupancy is not None
                else None
            ),
            "map_2d": (
                self.latest_rgb_map.summary()
                if self.latest_rgb_map is not None
                else None
            ),
            "loop_closure": loop_summary,
            "timing": timing_summary,
        }
        if self.config.do_save:
            self._write_text_atomic(
                self.config.output_dir / "summary.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
            )
            self._write_text_atomic(
                self.config.output_dir / "timing_summary.json",
                json.dumps(timing_summary, indent=2, sort_keys=True) + "\n",
            )
        print(
            f"[mapping] complete: {summary['frames_accepted']}/{summary['frames_processed']} "
            f"frames accepted, {summary['map_voxels']} voxels"
        )
        print(
            "[timing] finalize: "
            f"products={final_products_seconds:.6f}s "
            f"saving={final_saving_seconds:.6f}s "
            f"reports={final_report_seconds:.6f}s "
            f"loop_closure={loop_seconds:.6f}s "
            f"total={self.final_timing['finalization_work_total_seconds']:.6f}s"
        )
        print(f"[timing] overall work total={overall_seconds:.6f}s")
        return summary
