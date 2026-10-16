"""Input/path adapter for incremental mapping."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

from mapping.types import FrameData


class MapIO:
    """Infer all mapping inputs from ``fusion/<sequence>/fused``."""

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Incrementally register and fuse colored point clouds"
        )
        parser.add_argument(
            "--src", required=True, type=Path,
            help="Fused PCD directory, e.g. data/usegeo/fusion/test/fused",
        )
        parser.add_argument(
            "--dst", type=str,
            help="Mapping sequence/output name (default: source sequence name)",
        )
        parser.add_argument("--index", type=int, help="Only process this data.json index")
        parser.add_argument("--start", type=int, help="Start at this data.json index")
        parser.add_argument("--save", action="store_true", help="Save map checkpoints")
        parser.add_argument(
            "--position-mode", choices=("relative", "off"),
            help="Use pose_utm relative to frame 0, or rely on registration only",
        )
        parser.add_argument("--voxel-size", type=float, help="Map voxel size in meters")
        parser.add_argument(
            "--no-visual", action="store_true", help="Disable SIFT/RANSAC matching"
        )
        parser.add_argument(
            "--no-ground", action="store_true", help="Disable generated-ground constraint"
        )
        parser.add_argument(
            "--max-keyframes", type=int,
            help="Maximum prior keyframes tested visually per frame",
        )
        parser.add_argument("--seed", type=int, help="Deterministic RANSAC seed")
        parser.add_argument(
            "--visualize",
            action="store_true",
            help="Open the cumulative-map viewer; press A for each frame and Q to quit",
        )
        parser.add_argument(
            "--no-mesh", action="store_true", help="Do not write cumulative PLY meshes"
        )
        parser.add_argument(
            "--save-derived-steps",
            action="store_true",
            help=(
                "Also save mesh, occupancy, and RGB-map products after every frame "
                "(slower; cumulative PCD checkpoints are always saved with --save)"
            ),
        )
        parser.add_argument(
            "--no-occupancy",
            action="store_true",
            help="Do not generate the sparse probabilistic 3D occupancy grid",
        )
        parser.add_argument(
            "--occupancy-voxel-size",
            "--occupancy-resolution",
            dest="occupancy_voxel_size",
            type=float,
            help="Sparse 3D occupancy voxel size in meters",
        )
        parser.add_argument(
            "--occupancy-max-rays",
            type=int,
            help="Maximum unique endpoint rays integrated per frame",
        )
        parser.add_argument(
            "--no-map-2d",
            action="store_true",
            help="Do not write the orthographic RGB/mask map",
        )
        parser.add_argument(
            "--map-2d-resolution",
            type=float,
            help="Orthographic RGB map pixel size in meters",
        )
        parser.add_argument(
            "--no-diagnostics",
            action="store_true",
            help="Do not save visual match/overlay and geometry diagnostic files",
        )
        parser.add_argument(
            "--loop-closure",
            action="store_true",
            help="Also build optimized/ using verified loops and global pose optimization",
        )
        parser.add_argument(
            "--loop-search-radius",
            type=float,
            help="Maximum camera-center distance for loop candidates in meters",
        )
        self.args = parser.parse_args()

        self.source_dir = self.args.src.expanduser().resolve()
        if not self.source_dir.is_dir():
            parser.error(f"source directory not found: {self.source_dir}")
        try:
            self.dataset_dir = self.source_dir.parents[2]
        except IndexError:
            parser.error("--src must look like <dataset>/fusion/<sequence>/fused")
        self.source_sequence = self.source_dir.parent.name
        self.dst = self.args.dst or self.source_sequence

        config_path = self.dataset_dir / "config.yaml"
        data_path = self.dataset_dir / "data.json"
        if not config_path.is_file():
            parser.error(f"configuration not found: {config_path}")
        if not data_path.is_file():
            parser.error(f"frame metadata not found: {data_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle) or {}
        with data_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not rows:
            parser.error(f"expected a non-empty JSON list in {data_path}")
        self.rows = rows

        try:
            self.origin_utm = np.asarray(rows[0]["pose_utm"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"invalid first pose_utm in {data_path}: {exc}")
        if self.origin_utm.shape != (3,) or not np.isfinite(self.origin_utm).all():
            parser.error(f"first pose_utm must contain three finite values: {data_path}")

    def getDataRootDir(self) -> Path:
        return self.dataset_dir

    def getDstDir(self) -> str:
        return self.dst

    def getDoSave(self) -> bool:
        return bool(self.args.save)

    def getVisualize(self) -> bool:
        return bool(self.args.visualize)

    def mapping_options(self) -> Dict:
        options = dict(self.cfg.get("mapping", {}) or {})
        cli_values = {
            "position_mode": self.args.position_mode,
            "voxel_size": self.args.voxel_size,
            "max_keyframe_candidates": self.args.max_keyframes,
            "random_seed": self.args.seed,
            "occupancy_voxel_size": self.args.occupancy_voxel_size,
            "occupancy_max_rays_per_frame": self.args.occupancy_max_rays,
            "map_2d_resolution": self.args.map_2d_resolution,
            "loop_search_radius": self.args.loop_search_radius,
        }
        options.update({key: value for key, value in cli_values.items() if value is not None})
        if self.args.no_visual:
            options["use_visual"] = False
        if self.args.no_ground:
            options["use_ground"] = False
        if self.args.no_mesh:
            options["save_mesh"] = False
        if self.args.save_derived_steps:
            options["save_derived_steps"] = True
        if self.args.no_occupancy:
            options["save_occupancy"] = False
        if self.args.no_map_2d:
            options["save_map_2d"] = False
        if self.args.no_diagnostics:
            options["save_diagnostics"] = False
        if self.args.loop_closure:
            options["run_loop_closure"] = True
        return options

    def _selected_rows(self) -> Iterable[Dict]:
        rows = self.rows
        if self.args.index is not None:
            selected = [row for row in rows if int(row.get("index", -1)) == self.args.index]
            if not selected:
                raise ValueError(f"index {self.args.index} not found in data.json")
            return selected[:1]
        if self.args.start is not None:
            starts = [i for i, row in enumerate(rows) if int(row.get("index", -1)) == self.args.start]
            if not starts:
                raise ValueError(f"start index {self.args.start} not found in data.json")
            rows = rows[starts[0]:]
        return rows

    def _read_cloud(self, path: Path):
        if o3d is None:
            raise RuntimeError(
                "Open3D is required for mapping. Install the same open3d dependency "
                "used by projection and fusion."
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            cloud = o3d.io.read_point_cloud(
                str(path), remove_nan_points=False, remove_infinite_points=False
            )
        except TypeError:
            cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64).copy()
        if not len(points):
            raise IOError(f"point cloud is empty or unreadable: {path}")
        colors = None
        if cloud.has_colors() and len(cloud.colors) == len(points):
            colors = np.asarray(cloud.colors, dtype=np.float64).copy()
        return points, colors

    def _find_image(self, name: str) -> Optional[Path]:
        image_dir = self.dataset_dir / "projection" / self.source_sequence / "rgb"
        configured = str(self.cfg.get("rgb_extension", ".png"))
        suffixes = [".png", configured, ".jpg", ".jpeg", ".tif", ".tiff"]
        seen = set()
        for suffix in suffixes:
            suffix = suffix if suffix.startswith(".") else f".{suffix}"
            if suffix.lower() in seen:
                continue
            seen.add(suffix.lower())
            candidate = image_dir / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _read_image(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None or not self.mapping_options().get("use_visual", True):
            return None
        if cv2 is None:
            raise RuntimeError(
                "OpenCV with SIFT support is required for visual mapping. Install "
                "opencv-python (or run with --no-visual)."
            )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"image is unreadable: {path}")
        return image

    def load_row(self, row: Dict) -> FrameData:
        try:
            index = int(row["index"])
            name = str(row["name"])
            position = np.asarray(row["pose_utm"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid row in data.json: {exc}") from exc
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"frame {name} has an invalid pose_utm")

        source_path = self.source_dir / f"{name}.pcd"
        points, colors = self._read_cloud(source_path)
        finite_points = points[np.isfinite(points).all(axis=1)]
        if not len(finite_points):
            raise ValueError(f"frame {name} contains no finite 3D points")
        if float(np.median(np.linalg.norm(finite_points, axis=1))) > 10_000.0:
            raise ValueError(
                f"frame {name} appears to contain absolute/translated coordinates. "
                "Rerun projection and fusion without --on-video before mapping."
            )

        image_path = self._find_image(name)
        image = self._read_image(image_path)
        if image_path is None and self.mapping_options().get("use_visual", True):
            print(f"[mapping][warning] projected RGB not found for {name}; using geometry only")
        if colors is None and image is not None and len(points) == image.shape[0] * image.shape[1]:
            colors = image.reshape((-1, 3))[:, ::-1].astype(np.float64) / 255.0

        ground_path = self.source_dir.parent / "ground" / f"{name}.pcd"
        ground_points = None
        if self.mapping_options().get("use_ground", True):
            if ground_path.is_file():
                ground_points, _ = self._read_cloud(ground_path)
            else:
                print(f"[mapping][warning] generated ground PCD not found for {name}")

        return FrameData(
            index=index,
            name=name,
            position_utm=position,
            relative_position=position - self.origin_utm,
            points=points,
            colors=colors,
            image=image,
            ground_points=ground_points,
            source_path=str(source_path),
            image_path=str(image_path) if image_path else None,
            ground_path=str(ground_path) if ground_path.is_file() else None,
        )

    def load(self):
        for row in self._selected_rows():
            yield self.load_row(row)
