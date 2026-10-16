import math
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "lib"))

from mapping.config import MappingConfig
from mapping.geometry_viewer import load_geometry
from mapping.mapper import IncrementalMapper
from mapping.occupancy3d import FREE, KNOWN, OCCUPIED, OccupancyGrid3D
from mapping.products import (
    build_height_field_mesh,
    build_rgb_map_2d,
)
from mapping.registration import (
    cv2,
    ground_plane_alignment,
    o3d,
    ransac_rigid,
    transform_points,
)
from mapping.types import FrameData
from mapping.visualizer import MappingVisualizer
from mapping.voxel_map import VoxelMap


def transform_z(angle_deg, translation):
    angle = math.radians(angle_deg)
    transform = np.eye(4)
    transform[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = translation
    return transform


class RegistrationTests(unittest.TestCase):
    def test_rigid_ransac_rejects_outliers(self):
        rng = np.random.default_rng(11)
        source = rng.normal(size=(120, 3))
        expected = transform_z(7.0, [1.2, -0.7, 0.3])
        target = transform_points(source, expected)
        target += rng.normal(scale=0.003, size=target.shape)
        target[:35] = rng.uniform(-10.0, 10.0, size=(35, 3))

        actual, inliers, rmse = ransac_rigid(source, target, 0.03, 800, rng)
        self.assertIsNotNone(actual)
        self.assertGreaterEqual(int(inliers.sum()), 84)
        self.assertLess(rmse, 0.01)
        np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=0.005)
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=0.01)

    def test_ground_alignment_only_corrects_plane_normal_and_height(self):
        x, y = np.meshgrid(np.linspace(-8, 8, 30), np.linspace(-5, 5, 20))
        target = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        source = target.copy()
        source[:, 2] = 0.08 * source[:, 0] + 0.7

        transform, report = ground_plane_alignment(
            source, target, np.eye(4), 50, 10.0, 2.0
        )
        corrected = transform_points(source, transform)
        self.assertTrue(report["used"])
        self.assertLess(float(np.sqrt(np.mean(corrected[:, 2] ** 2))), 1e-6)


class VoxelMapTests(unittest.TestCase):
    def test_voxel_fusion_is_deterministic_and_bounded(self):
        voxel_map = VoxelMap(1.0, max_weight=2.0)
        first = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.2, 0.1, 0.1]])
        colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        inserted = voxel_map.integrate(first, colors, 1.0, 0)
        self.assertEqual(inserted, 2)
        self.assertEqual(len(voxel_map), 2)

        voxel_map.integrate(first + [0.1, 0.0, 0.0], colors, 1.0, 1)
        points, fused_colors = voxel_map.to_arrays()
        self.assertEqual(points.shape, (2, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(((fused_colors >= 0.0) & (fused_colors <= 1.0)).all())
        self.assertTrue(
            (voxel_map.state_arrays()["map_geom_weights"] <= 2.0).all()
        )


class DerivedProductTests(unittest.TestCase):
    def test_height_mesh_has_colored_triangles(self):
        x, y = np.meshgrid(np.arange(4, dtype=float) + 0.1, np.arange(3, dtype=float) + 0.1)
        points = np.column_stack((x.ravel(), y.ravel(), (0.1 * x + 0.2 * y).ravel()))
        colors = np.tile([0.2, 0.4, 0.6], (len(points), 1))
        mesh = build_height_field_mesh(points, colors, 1.0, 1.0)
        self.assertEqual(len(mesh.vertices), 12)
        self.assertEqual(len(mesh.triangles), 12)
        self.assertTrue((mesh.colors == colors[0]).all())

    def test_3d_occupancy_integrates_free_known_and_occupied(self):
        grid = OccupancyGrid3D(1.0, max_rays_per_frame=20, max_ray_length=20.0)
        update = grid.integrate(
            np.array([0.1, 0.1, 0.1]),
            np.array([[3.2, 0.1, 0.1]]),
            confidence=1.0,
            frame_index=0,
        )
        keys, states, _, _, _ = grid.to_arrays()
        by_key = {tuple(key): state for key, state in zip(keys, states)}
        self.assertGreater(update["free_updates"], 0)
        self.assertEqual(by_key[(3, 0, 0)], OCCUPIED)
        self.assertEqual(by_key[(1, 0, 0)], FREE)

        uncertain = OccupancyGrid3D(1.0)
        uncertain.integrate(
            np.zeros(3), np.array([[2.2, 0.0, 0.0]]), 0.1, 1
        )
        uncertain_states = uncertain.to_arrays()[1]
        self.assertIn(KNOWN, uncertain_states)

    def test_rgb_map_selects_top_surface_and_mask(self):
        points = np.array(
            [[0.2, 0.2, 0.0], [0.3, 0.3, 2.0], [1.2, 0.2, 1.0]],
            dtype=float,
        )
        colors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        rgb_map = build_rgb_map_2d(points, colors, 1.0, padding=0.0)
        pixel = rgb_map.world_to_pixel([[0.2, 0.2]])[0]
        np.testing.assert_array_equal(
            rgb_map.image_rgb[pixel[1], pixel[0]], [0, 255, 0]
        )
        self.assertEqual(rgb_map.mask[pixel[1], pixel[0]], 255)

    def test_disabled_visualizer_never_blocks(self):
        viewer = MappingVisualizer(False)
        self.assertTrue(viewer.wait_for_advance(0, "frame"))
        viewer.update(np.empty((0, 3)), np.empty((0, 3)), "test")
        viewer.finish()


@unittest.skipIf(o3d is None, "Open3D is not installed")
class IncrementalMapperTests(unittest.TestCase):
    def test_two_frame_registration_and_accumulation(self):
        x, y = np.meshgrid(np.linspace(-12, 12, 55), np.linspace(-9, 9, 45))
        world = np.column_stack(
            (x.ravel(), y.ravel(), (0.6 * np.sin(x / 3.0) + 0.3 * np.cos(y / 2.0)).ravel())
        )
        colors = np.column_stack(
            (
                (x.ravel() - x.min()) / np.ptp(x),
                (y.ravel() - y.min()) / np.ptp(y),
                np.full(x.size, 0.5),
            )
        )
        relative = np.array([3.0, -1.0, 0.2])
        sensor_error = np.array([0.35, -0.20, 0.10])

        frame0 = FrameData(
            0, "frame0", np.zeros(3), np.zeros(3), world, colors, None, None, "frame0.pcd"
        )
        frame1 = FrameData(
            1,
            "frame1",
            relative,
            relative,
            world - relative + sensor_error,
            colors,
            None,
            None,
            "frame1.pcd",
        )
        config = MappingConfig(
            root_dir=PROJECT_ROOT,
            dst_dir="synthetic",
            do_save=False,
            use_visual=False,
            use_ground=False,
            voxel_size=0.45,
            min_icp_fitness=0.05,
        )
        mapper = IncrementalMapper(config)
        first_result = mapper.process(frame0)
        second_result = mapper.process(frame1)
        summary = mapper.finalize()

        self.assertTrue(first_result.accepted)
        self.assertTrue(second_result.accepted, second_result.message)
        self.assertGreater(second_result.overlap_after, 0.70)
        self.assertLess(second_result.rmse_after, 0.30)
        self.assertEqual(summary["frames_accepted"], 2)
        self.assertGreater(len(mapper.map), 500)


@unittest.skipIf(o3d is None or cv2 is None, "Open3D/OpenCV are not installed")
class CommandLineSmokeTests(unittest.TestCase):
    @staticmethod
    def _write_cloud(path, points, colors=None):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        if not o3d.io.write_point_cloud(str(path), cloud):
            raise IOError(path)

    def test_launch_map_end_to_end_with_visual_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "usegeo"
            fused_dir = dataset / "fusion" / "test" / "fused"
            ground_dir = dataset / "fusion" / "test" / "ground"
            image_dir = dataset / "projection" / "test" / "rgb"
            fused_dir.mkdir(parents=True)
            ground_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            height, width = 72, 96
            x, y = np.meshgrid(np.linspace(-12, 12, width), np.linspace(-9, 9, height))
            world = np.column_stack(
                (
                    x.ravel(),
                    y.ravel(),
                    (-8.0 + 0.5 * np.sin(x / 2.5) + 0.2 * np.cos(y / 2)).ravel(),
                )
            )
            rng = np.random.default_rng(23)
            image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            colors = image.reshape((-1, 3))[:, ::-1].astype(float) / 255.0
            relative_positions = np.array(
                [
                    [0.000, 0.000, 0.000],
                    [-8.909, 2.626, -0.072],
                    [-12.139, 4.018, -0.297],
                    [-10.697, 7.320, -0.559],
                    [-6.405, 13.750, -0.856],
                    [-0.118, 21.413, -1.160],
                    [6.903, 29.745, -1.464],
                    [14.520, 38.360, -1.771],
                ]
            )
            names = [
                f"2021-04-23_13-17-{second:02d}_S2223314_DxO"
                for second in range(12, 28, 2)
            ]
            errors = np.column_stack(
                (
                    np.linspace(0.0, 0.30, len(names)),
                    np.linspace(0.0, -0.18, len(names)),
                    np.linspace(0.0, 0.08, len(names)),
                )
            )
            local_clouds = [
                world - relative + error
                for relative, error in zip(relative_positions, errors)
            ]

            ground_xy = world[:, :2]
            ground_world = np.column_stack((ground_xy, np.full(len(ground_xy), -8.0)))
            ground_clouds = [
                ground_world - relative + error
                for relative, error in zip(relative_positions, errors)
            ]
            for name, points, ground in zip(names, local_clouds, ground_clouds):
                self._write_cloud(fused_dir / f"{name}.pcd", points, colors)
                self._write_cloud(ground_dir / f"{name}.pcd", ground)
                self.assertTrue(cv2.imwrite(str(image_dir / f"{name}.png"), image))

            origin = np.array([498000.0, 4379000.0, 190.0])
            rows = [
                {
                    "index": index,
                    "name": name,
                    "pose_utm": (origin + relative).tolist(),
                }
                for index, (name, relative) in enumerate(zip(names, relative_positions))
            ]
            (dataset / "data.json").write_text(json.dumps(rows), encoding="utf-8")
            (dataset / "config.yaml").write_text(
                "hfov_deg: 81.14\nkinematics: opk\nrgb_extension: .png\nzone: 32N_UTM\n",
                encoding="utf-8",
            )

            environment = dict(os.environ)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "launch" / "map.py"),
                "--src",
                str(fused_dir),
                "--save",
                "--loop-closure",
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=240,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = dataset / "mapping" / "test"
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            report = json.loads(
                (output / "reports" / f"000007_{names[-1]}.json").read_text(encoding="utf-8")
            )
            mapped = o3d.io.read_point_cloud(str(output / "map.pcd"))
            mesh = o3d.io.read_triangle_mesh(str(output / "map_mesh.ply"))
            occupancy = np.load(output / "occupancy" / "occupancy_grid_3d.npz")
            occupancy_cloud = o3d.io.read_point_cloud(
                str(output / "occupancy" / "occupancy_grid_3d.pcd")
            )
            comparison = json.loads(
                (output / "optimized" / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["frames_accepted"], 8)
            self.assertTrue(report["accepted"], report["message"])
            self.assertTrue(report["visual"]["success"])
            self.assertGreater(report["mesh"]["triangles"], 100)
            self.assertGreater(report["occupancy"]["observed_voxels"], 1)
            self.assertEqual(len(list((output / "steps").glob("*.pcd"))), 8)
            self.assertFalse((output / "mesh_steps").exists())
            self.assertFalse((output / "occupancy" / "steps").exists())
            self.assertFalse((output / "map_2d" / "steps").exists())
            self.assertGreater(len(mapped.points), 500)
            self.assertGreater(len(mesh.triangles), 100)
            self.assertTrue(
                set(np.unique(occupancy["occupancy3d_states"])).issubset({0, 50, 100})
            )
            self.assertTrue((output / "occupancy" / "occupancy_grid_3d.pcd").is_file())
            self.assertGreater(len(occupancy_cloud.points), 1)
            self.assertTrue(occupancy_cloud.has_colors())
            self.assertTrue((output / "map_2d" / "map_rgb.png").is_file())
            self.assertTrue((output / "map_2d" / "map_mask.png").is_file())
            self.assertTrue((output / "map_2d" / "map_trajectory.png").is_file())
            self.assertTrue((output / "trajectory.ply").is_file())
            timings = [
                json.loads(line)
                for line in (output / "timings.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            timing_summary = json.loads(
                (output / "timing_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(timings), 8)
            self.assertEqual(timing_summary["frames"]["frame_count"], 8)
            self.assertGreater(timing_summary["overall_work_total_seconds"], 0.0)
            self.assertTrue(all(item["mapping_seconds"] > 0.0 for item in timings))
            self.assertTrue(all(item["checkpoint_saving_seconds"] > 0.0 for item in timings))
            self.assertEqual(completed.stdout.count("[timing] frame "), 8)
            self.assertEqual(len(list((output / "reports" / "geometry").glob("*_geometry.json"))), 8)
            visual_matches = list(
                (output / "reports" / "visual").glob("*_matches.png")
            )
            self.assertGreaterEqual(len(visual_matches), 7)
            match_image = cv2.imread(str(visual_matches[0]), cv2.IMREAD_COLOR)
            self.assertIsNotNone(match_image)
            self.assertGreater(match_image.shape[1], width)
            mask_overlays = list(
                (output / "reports" / "visual").glob("*_mask_overlay.png")
            )
            self.assertGreaterEqual(len(mask_overlays), 7)
            self.assertGreater(
                np.count_nonzero(cv2.imread(str(mask_overlays[0]), cv2.IMREAD_COLOR)),
                0,
            )
            self.assertTrue((output / "optimized" / "map.pcd").is_file())
            self.assertTrue((output / "optimized" / "occupancy_grid_3d.pcd").is_file())
            self.assertTrue((output / "optimized" / "pose_graph.json").is_file())
            self.assertGreaterEqual(comparison["candidate_loops"], 1)
            self.assertGreaterEqual(comparison["accepted_loops"], 1)
            self.assertTrue(comparison["loop_residual_improved"])

            voxel_geometry, voxel_mode = load_geometry(
                output / "occupancy" / "occupancy_grid_3d.pcd"
            )
            mesh_geometry, mesh_mode = load_geometry(output / "map_mesh.ply")
            trajectory_geometry, trajectory_mode = load_geometry(
                output / "trajectory.ply"
            )
            self.assertEqual(voxel_mode, "voxel grid (1 m cubes)")
            self.assertGreater(len(voxel_geometry[0].get_voxels()), 1)
            self.assertEqual(mesh_mode, "triangle mesh")
            self.assertGreater(len(mesh_geometry[0].triangles), 100)
            self.assertEqual(trajectory_mode, "trajectory lines")
            self.assertGreater(len(trajectory_geometry[0].lines), 0)


if __name__ == "__main__":
    unittest.main()
