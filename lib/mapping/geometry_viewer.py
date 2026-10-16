"""Small Open3D viewer for mapping PCD, occupancy-voxel, and PLY outputs."""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def _require_open3d():
    if o3d is None:
        raise RuntimeError(
            "Open3D is required for the 3D viewer. Install requirements-mapping.txt."
        )


def occupancy_voxel_size(path: Path) -> Optional[float]:
    """Read the voxel size from the JSON stored next to an occupancy PCD."""
    metadata_path = Path(path).with_suffix(".json")
    if not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        value = float(metadata["voxel_size_m"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _read_trajectory_lines(path: Path):
    """Load this mapper's ASCII edge PLY as a LineSet, otherwise return None."""
    try:
        with Path(path).open("r", encoding="ascii") as handle:
            header = []
            while True:
                line = handle.readline()
                if not line:
                    return None
                header.append(line.strip())
                if line.strip() == "end_header":
                    break
            if "format ascii 1.0" not in header:
                return None

            vertex_count = 0
            edge_count = 0
            for line in header:
                fields = line.split()
                if fields[:2] == ["element", "vertex"] and len(fields) == 3:
                    vertex_count = int(fields[2])
                elif fields[:2] == ["element", "edge"] and len(fields) == 3:
                    edge_count = int(fields[2])
            if vertex_count < 1 or edge_count < 1:
                return None

            vertices = []
            vertex_colors = []
            for _ in range(vertex_count):
                fields = handle.readline().split()
                if len(fields) < 3:
                    return None
                vertices.append([float(value) for value in fields[:3]])
                if len(fields) >= 6:
                    vertex_colors.append([int(value) / 255.0 for value in fields[3:6]])
            edges = []
            for _ in range(edge_count):
                fields = handle.readline().split()
                if len(fields) < 2:
                    return None
                edges.append([int(fields[0]), int(fields[1])])
    except (OSError, UnicodeDecodeError, ValueError):
        return None

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    if len(vertex_colors) == len(vertices):
        colors = np.asarray(vertex_colors, dtype=np.float64)
        line_colors = 0.5 * (colors[np.asarray(edges)[:, 0]] + colors[np.asarray(edges)[:, 1]])
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set


def load_geometry(
    path: Path,
    render_voxels: Optional[bool] = None,
    voxel_size: Optional[float] = None,
) -> Tuple[List[object], str]:
    """Load one supported file and return Open3D geometries plus a mode label.

    ``render_voxels=None`` auto-detects mapping occupancy PCDs from their sibling
    JSON.  ``True`` forces cubes and requires either ``voxel_size`` or metadata.
    """
    _require_open3d()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix not in {".pcd", ".ply"}:
        raise ValueError(f"unsupported 3D file '{path.suffix}'; expected .pcd or .ply")

    if suffix == ".ply":
        line_set = _read_trajectory_lines(path)
        if line_set is not None:
            return [line_set], "trajectory lines"
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh is not None and len(mesh.vertices) and len(mesh.triangles):
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            return [mesh], "triangle mesh"

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud is None or not len(cloud.points):
        raise IOError(f"file contains no readable points or triangles: {path}")

    metadata_size = occupancy_voxel_size(path)
    use_voxels = metadata_size is not None if render_voxels is None else render_voxels
    if use_voxels:
        size = voxel_size if voxel_size is not None else metadata_size
        if size is None or not np.isfinite(size) or size <= 0:
            raise ValueError("--voxels requires --voxel-size or sibling occupancy JSON")
        grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud, float(size))
        return [grid], f"voxel grid ({float(size):g} m cubes)"
    return [cloud], "point cloud"


def show_files(
    paths: List[Path],
    render_voxels: Optional[bool] = None,
    voxel_size: Optional[float] = None,
    show_axes: bool = True,
):
    """Open one window containing all supplied mapping geometries."""
    _require_open3d()
    geometries = []
    for path in paths:
        loaded, mode = load_geometry(path, render_voxels, voxel_size)
        geometries.extend(loaded)
        print(f"[view3d] {Path(path)}: {mode}")
    if show_axes:
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=max(1.0, float(voxel_size or 1.0))
            )
        )
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Mapping PCD/PLY viewer",
        width=1200,
        height=800,
    )

