"""A small deterministic voxel/surfel accumulator used by the mapper."""

from typing import Dict, Optional, Tuple

import numpy as np


def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    colors = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    return np.where(
        colors <= 0.04045,
        colors / 12.92,
        ((colors + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    colors = np.maximum(np.asarray(colors, dtype=np.float64), 0.0)
    return np.clip(
        np.where(
            colors <= 0.0031308,
            12.92 * colors,
            1.055 * np.power(colors, 1.0 / 2.4) - 0.055,
        ),
        0.0,
        1.0,
    )


class VoxelMap:
    """One weighted geometric/color surfel per voxel.

    A frame contributes at most one observation to a voxel.  This prevents a
    dense frame from overwhelming observations from other viewpoints.
    """

    def __init__(
        self,
        voxel_size: float,
        max_weight: float = 30.0,
        color_sigma: float = 0.20,
        color_reject_distance: float = 0.45,
    ):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.max_weight = float(max_weight)
        self.color_sigma = max(float(color_sigma), 1e-6)
        self.color_reject_distance = float(color_reject_distance)

        self._lookup: Dict[Tuple[int, int, int], int] = {}
        self._keys = []
        self._positions = []
        self._geom_weights = []
        self._colors_linear = []
        self._color_weights = []
        self._observations = []
        self._last_seen = []

    def __len__(self) -> int:
        return len(self._positions)

    def integrate(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        confidence: float,
        frame_index: int,
    ) -> int:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3); got {points.shape}")

        valid = np.isfinite(points).all(axis=1)
        points = points[valid]
        if not len(points):
            return 0

        have_colors = colors is not None
        if have_colors:
            colors = np.asarray(colors, dtype=np.float64)
            if colors.shape != valid.shape + (3,):
                raise ValueError(
                    f"colors must have shape ({len(valid)}, 3); got {colors.shape}"
                )
            colors = colors[valid]
            if np.nanmax(colors) > 1.0:
                colors = colors / 255.0
            color_valid = np.isfinite(colors).all(axis=1)
            colors = np.clip(np.where(color_valid[:, None], colors, 0.0), 0.0, 1.0)
            colors_linear = srgb_to_linear(colors)
        else:
            color_valid = np.zeros(len(points), dtype=bool)
            colors_linear = np.zeros_like(points)

        confidence = float(np.clip(confidence, 0.01, 1.0))
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        n_unique = len(unique_keys)

        counts = np.bincount(inverse, minlength=n_unique).astype(np.float64)
        position_sums = np.zeros((n_unique, 3), dtype=np.float64)
        np.add.at(position_sums, inverse, points)
        position_means = position_sums / counts[:, None]

        color_counts = np.bincount(
            inverse, weights=color_valid.astype(np.float64), minlength=n_unique
        )
        color_sums = np.zeros((n_unique, 3), dtype=np.float64)
        np.add.at(color_sums, inverse, colors_linear * color_valid[:, None])
        color_means = color_sums / np.maximum(color_counts[:, None], 1.0)

        inserted = 0
        for j, key_arr in enumerate(unique_keys):
            key = tuple(int(v) for v in key_arr)
            observed_position = position_means[j]
            map_index = self._lookup.get(key)

            if map_index is None:
                map_index = len(self._positions)
                self._lookup[key] = map_index
                self._keys.append(key)
                self._positions.append(observed_position.copy())
                self._geom_weights.append(confidence)
                if color_counts[j] > 0:
                    self._colors_linear.append(color_means[j].copy())
                    self._color_weights.append(confidence)
                else:
                    self._colors_linear.append(np.zeros(3, dtype=np.float64))
                    self._color_weights.append(0.0)
                self._observations.append(1)
                self._last_seen.append(int(frame_index))
                inserted += 1
                continue

            old_position = self._positions[map_index]
            old_weight = self._geom_weights[map_index]
            # Same-voxel observations near the current surfel get full weight;
            # observations near a voxel corner are blended more cautiously.
            displacement = float(np.linalg.norm(observed_position - old_position))
            geom_factor = np.exp(
                -0.5 * (displacement / max(0.75 * self.voxel_size, 1e-9)) ** 2
            )
            new_weight = confidence * float(np.clip(geom_factor, 0.05, 1.0))
            denom = old_weight + new_weight
            self._positions[map_index] = (
                old_weight * old_position + new_weight * observed_position
            ) / denom
            self._geom_weights[map_index] = min(self.max_weight, denom)

            if color_counts[j] > 0:
                old_color_weight = self._color_weights[map_index]
                observed_color = color_means[j]
                color_factor = 1.0
                if old_color_weight > 0:
                    old_srgb = linear_to_srgb(self._colors_linear[map_index])
                    observed_srgb = linear_to_srgb(observed_color)
                    color_delta = float(np.linalg.norm(observed_srgb - old_srgb))
                    color_factor = np.exp(-0.5 * (color_delta / self.color_sigma) ** 2)
                    if (
                        self._observations[map_index] >= 2
                        and color_delta > self.color_reject_distance
                    ):
                        color_factor = 0.0

                color_weight = confidence * float(color_factor)
                if color_weight > 0:
                    color_denom = old_color_weight + color_weight
                    self._colors_linear[map_index] = (
                        old_color_weight * self._colors_linear[map_index]
                        + color_weight * observed_color
                    ) / color_denom
                    self._color_weights[map_index] = min(
                        self.max_weight, color_denom
                    )

            self._observations[map_index] += 1
            self._last_seen[map_index] = int(frame_index)

        return inserted

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not len(self):
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        points = np.asarray(self._positions, dtype=np.float64)
        colors_linear = np.asarray(self._colors_linear, dtype=np.float64)
        colors = linear_to_srgb(colors_linear)
        missing = np.asarray(self._color_weights, dtype=np.float64) <= 0
        colors[missing] = 0.5
        return points, colors

    def extract_bounds(
        self, minimum: np.ndarray, maximum: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        points, colors = self.to_arrays()
        if not len(points):
            return points, colors
        minimum = np.asarray(minimum, dtype=np.float64)
        maximum = np.asarray(maximum, dtype=np.float64)
        keep = np.all((points >= minimum) & (points <= maximum), axis=1)
        return points[keep], colors[keep]

    def state_arrays(self, prefix: str = "map") -> Dict[str, np.ndarray]:
        points, colors = self.to_arrays()
        keys = np.asarray(self._keys, dtype=np.int64).reshape((-1, 3))
        return {
            f"{prefix}_voxel_size": np.asarray(self.voxel_size),
            f"{prefix}_keys": keys,
            f"{prefix}_points": points,
            f"{prefix}_colors": colors,
            f"{prefix}_geom_weights": np.asarray(self._geom_weights, dtype=np.float64),
            f"{prefix}_color_weights": np.asarray(self._color_weights, dtype=np.float64),
            f"{prefix}_observations": np.asarray(self._observations, dtype=np.int32),
            f"{prefix}_last_seen": np.asarray(self._last_seen, dtype=np.int32),
        }
