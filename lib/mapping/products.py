"""Colored surface mesh and orthographic RGB map products."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    colors: np.ndarray
    triangles: np.ndarray

    @property
    def is_empty(self) -> bool:
        return len(self.vertices) == 0 or len(self.triangles) == 0

    def summary(self) -> Dict:
        return {
            "vertices": int(len(self.vertices)),
            "triangles": int(len(self.triangles)),
        }


@dataclass
class RgbMap2D:
    """North-up orthographic image of the highest observed colored surface."""

    image_rgb: np.ndarray
    mask: np.ndarray
    elevation: np.ndarray
    origin_xy: np.ndarray
    resolution: float

    def summary(self) -> Dict:
        return {
            "width": int(self.image_rgb.shape[1]),
            "height": int(self.image_rgb.shape[0]),
            "resolution_m": float(self.resolution),
            "origin_xy": self.origin_xy.tolist(),
            "observed_pixels": int(np.count_nonzero(self.mask)),
        }

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape((-1, 2))
        cell = np.floor((xy - self.origin_xy) / self.resolution).astype(np.int64)
        pixels = np.empty_like(cell)
        pixels[:, 0] = cell[:, 0]
        pixels[:, 1] = self.image_rgb.shape[0] - 1 - cell[:, 1]
        return pixels


def _empty_mesh() -> SurfaceMesh:
    return SurfaceMesh(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int32),
    )


def build_height_field_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    grid_size: float,
    max_height_jump: float,
    max_edge_factor: float = 2.5,
) -> SurfaceMesh:
    """Triangulate the top observed surface on a regular XY grid."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or grid_size <= 0:
        return _empty_mesh()
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) < 3:
        return _empty_mesh()

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.shape == valid.shape + (3,):
            colors = np.clip(colors[valid], 0.0, 1.0)
        else:
            colors = None
    if colors is None:
        colors = np.full((len(points), 3), 0.5, dtype=np.float64)

    keys = np.floor(points[:, :2] / float(grid_size)).astype(np.int64)
    order = np.lexsort((-points[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    selected = order[first]
    vertices = points[selected]
    vertex_colors = colors[selected]
    vertex_keys = keys[selected]
    lookup = {tuple(key): i for i, key in enumerate(vertex_keys)}

    triangles = []
    max_edge = float(max_edge_factor) * float(grid_size)

    def add_triangle(indices):
        triangle_points = vertices[np.asarray(indices)]
        edges = (
            np.linalg.norm(triangle_points[1] - triangle_points[0]),
            np.linalg.norm(triangle_points[2] - triangle_points[1]),
            np.linalg.norm(triangle_points[0] - triangle_points[2]),
        )
        if max(edges) > max_edge:
            return
        if float(np.ptp(triangle_points[:, 2])) > max_height_jump:
            return
        if np.linalg.norm(
            np.cross(
                triangle_points[1] - triangle_points[0],
                triangle_points[2] - triangle_points[0],
            )
        ) <= 1e-10:
            return
        triangles.append(indices)

    for key, lower_left in lookup.items():
        x_key, y_key = key
        lower_right = lookup.get((x_key + 1, y_key))
        upper_left = lookup.get((x_key, y_key + 1))
        upper_right = lookup.get((x_key + 1, y_key + 1))
        if lower_right is not None and upper_left is not None:
            add_triangle((lower_left, lower_right, upper_left))
        if lower_right is not None and upper_left is not None and upper_right is not None:
            add_triangle((lower_right, upper_right, upper_left))

    if not triangles:
        return _empty_mesh()
    triangles = np.asarray(triangles, dtype=np.int32)
    used = np.unique(triangles)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return SurfaceMesh(vertices[used], vertex_colors[used], remap[triangles])


def build_rgb_map_2d(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    resolution: float,
    padding: float = 1.0,
    max_pixels: int = 16_000_000,
) -> RgbMap2D:
    """Rasterize the highest colored surface in every XY cell."""
    if resolution <= 0:
        raise ValueError("2D RGB map resolution must be positive")
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        colors = colors[valid] if colors.shape == valid.shape + (3,) else None
    if colors is None:
        colors = np.full((len(points), 3), 0.5, dtype=np.float64)
    colors = np.clip(colors, 0.0, 1.0)

    if not len(points):
        return RgbMap2D(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            np.full((1, 1), np.nan, dtype=np.float32),
            np.zeros(2, dtype=np.float64),
            float(resolution),
        )

    low = np.floor((np.min(points[:, :2], axis=0) - padding) / resolution) * resolution
    high = np.ceil((np.max(points[:, :2], axis=0) + padding) / resolution) * resolution
    size = np.maximum(np.rint((high - low) / resolution).astype(np.int64) + 1, 1)
    width, height = int(size[0]), int(size[1])
    if width * height > int(max_pixels):
        raise ValueError(
            f"2D RGB map would contain {width * height:,} pixels; increase "
            "map_2d_resolution or map_2d_max_pixels"
        )

    cells = np.floor((points[:, :2] - low) / resolution).astype(np.int64)
    inside = (
        (cells[:, 0] >= 0) & (cells[:, 0] < width)
        & (cells[:, 1] >= 0) & (cells[:, 1] < height)
    )
    points, colors, cells = points[inside], colors[inside], cells[inside]
    flat = cells[:, 1] * width + cells[:, 0]
    order = np.lexsort((-points[:, 2], flat))
    sorted_flat = flat[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_flat[1:] != sorted_flat[:-1]
    chosen = order[first]
    x_cell, y_cell = cells[chosen, 0], cells[chosen, 1]
    rows = height - 1 - y_cell

    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    elevation = np.full((height, width), np.nan, dtype=np.float32)
    image[rows, x_cell] = np.rint(colors[chosen] * 255.0).astype(np.uint8)
    mask[rows, x_cell] = 255
    elevation[rows, x_cell] = points[chosen, 2].astype(np.float32)
    return RgbMap2D(image, mask, elevation, low, float(resolution))
