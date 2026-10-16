"""Sparse probabilistic 3D occupancy grid built from camera-to-surface rays."""

from typing import Dict, Optional, Tuple

import numpy as np


UNKNOWN = np.int8(-1)  # Implicit: no stored observation.
FREE = np.int8(0)
KNOWN = np.int8(50)  # Observed, but probability is not decisive yet.
OCCUPIED = np.int8(100)

FREE_RGBA = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
KNOWN_RGBA = np.array([1.0, 0.72, 0.40, 0.75], dtype=np.float32)
OCCUPIED_RGBA = np.array([1.0, 0.55, 0.55, 1.0], dtype=np.float32)


class OccupancyGrid3D:
    """A sparse log-odds grid with implicit unknown voxels.

    Each frame contributes at most one free or occupied observation to a voxel,
    preventing dense images from overwhelming evidence from other viewpoints.
    """

    def __init__(
        self,
        voxel_size: float,
        free_log_odds: float = -0.40,
        occupied_log_odds: float = 0.85,
        free_threshold: float = -0.20,
        occupied_threshold: float = 0.40,
        clamp_min: float = -2.0,
        clamp_max: float = 3.5,
        max_rays_per_frame: int = 5_000,
        max_ray_length: float = 250.0,
        ray_step_factor: float = 1.0,
    ):
        if voxel_size <= 0:
            raise ValueError("3D occupancy voxel_size must be positive")
        if not 0 < ray_step_factor <= 1:
            raise ValueError("ray_step_factor must be in (0, 1]")
        self.voxel_size = float(voxel_size)
        self.free_log_odds = float(free_log_odds)
        self.occupied_log_odds = float(occupied_log_odds)
        self.free_threshold = float(free_threshold)
        self.occupied_threshold = float(occupied_threshold)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.max_rays_per_frame = int(max_rays_per_frame)
        self.max_ray_length = float(max_ray_length)
        self.ray_step_factor = float(ray_step_factor)

        self._log_odds: Dict[Tuple[int, int, int], float] = {}
        self._observations: Dict[Tuple[int, int, int], int] = {}
        self._last_seen: Dict[Tuple[int, int, int], int] = {}
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None

    def __len__(self):
        return len(self._log_odds)

    def _key(self, point: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor(point / self.voxel_size).astype(np.int64).tolist())

    def _update_bounds(self, keys: np.ndarray):
        if not len(keys):
            return
        minimum = np.min(keys, axis=0).astype(np.int64)
        maximum = np.max(keys, axis=0).astype(np.int64)
        if self._bounds_min is None:
            self._bounds_min = minimum
            self._bounds_max = maximum
        else:
            self._bounds_min = np.minimum(self._bounds_min, minimum)
            self._bounds_max = np.maximum(self._bounds_max, maximum)

    def integrate(
        self,
        camera_origin: np.ndarray,
        endpoints: np.ndarray,
        confidence: float,
        frame_index: int,
    ) -> Dict:
        camera = np.asarray(camera_origin, dtype=np.float64).reshape(3)
        endpoints = np.asarray(endpoints, dtype=np.float64).reshape((-1, 3))
        endpoints = endpoints[np.isfinite(endpoints).all(axis=1)]
        if not np.isfinite(camera).all() or not len(endpoints):
            return {"rays": 0, "free_updates": 0, "occupied_updates": 0}

        vectors = endpoints - camera
        distances = np.linalg.norm(vectors, axis=1)
        valid = (distances > self.voxel_size * 0.5) & (distances <= self.max_ray_length)
        endpoints = endpoints[valid]
        distances = distances[valid]
        if not len(endpoints):
            return {"rays": 0, "free_updates": 0, "occupied_updates": 0}

        endpoint_keys = np.floor(endpoints / self.voxel_size).astype(np.int64)
        # One ray per endpoint voxel is sufficient for a voxel occupancy update.
        _, selected = np.unique(endpoint_keys, axis=0, return_index=True)
        selected.sort()
        if len(selected) > self.max_rays_per_frame:
            keep = np.linspace(
                0, len(selected) - 1, self.max_rays_per_frame, dtype=np.int64
            )
            selected = selected[keep]
        endpoints = endpoints[selected]
        distances = distances[selected]
        endpoint_keys = endpoint_keys[selected]

        occupied_keys = {tuple(int(value) for value in key) for key in endpoint_keys}
        free_keys = set()
        sample_step = self.voxel_size * self.ray_step_factor
        for endpoint, distance, endpoint_key in zip(endpoints, distances, endpoint_keys):
            steps = max(2, int(np.ceil(distance / sample_step)))
            fractions = np.arange(steps, dtype=np.float64) / float(steps)
            samples = camera + fractions[:, None] * (endpoint - camera)
            keys = np.floor(samples / self.voxel_size).astype(np.int64)
            distinct = np.ones(len(keys), dtype=bool)
            distinct[1:] = np.any(keys[1:] != keys[:-1], axis=1)
            for key in keys[distinct]:
                key_tuple = tuple(int(value) for value in key)
                if key_tuple != tuple(int(value) for value in endpoint_key):
                    free_keys.add(key_tuple)

        # An observed surface endpoint wins over free traversal in the same frame.
        free_keys.difference_update(occupied_keys)
        confidence = float(np.clip(confidence, 0.05, 1.0))

        for key in free_keys:
            value = self._log_odds.get(key, 0.0) + confidence * self.free_log_odds
            self._log_odds[key] = float(np.clip(value, self.clamp_min, self.clamp_max))
            self._observations[key] = self._observations.get(key, 0) + 1
            self._last_seen[key] = int(frame_index)
        for key in occupied_keys:
            value = self._log_odds.get(key, 0.0) + confidence * self.occupied_log_odds
            self._log_odds[key] = float(np.clip(value, self.clamp_min, self.clamp_max))
            self._observations[key] = self._observations.get(key, 0) + 1
            self._last_seen[key] = int(frame_index)

        all_keys = np.asarray(list(free_keys | occupied_keys), dtype=np.int64).reshape((-1, 3))
        self._update_bounds(all_keys)
        return {
            "rays": int(len(endpoints)),
            "free_updates": int(len(free_keys)),
            "occupied_updates": int(len(occupied_keys)),
        }

    def classify_log_odds(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        states = np.full(values.shape, KNOWN, dtype=np.int8)
        states[values <= self.free_threshold] = FREE
        states[values >= self.occupied_threshold] = OCCUPIED
        return states

    def to_arrays(self):
        if not self._log_odds:
            return (
                np.empty((0, 3), dtype=np.int64),
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )
        keys = np.asarray(sorted(self._log_odds), dtype=np.int64)
        log_odds = np.asarray([self._log_odds[tuple(key)] for key in keys], dtype=np.float32)
        states = self.classify_log_odds(log_odds)
        observations = np.asarray(
            [self._observations[tuple(key)] for key in keys], dtype=np.int32
        )
        last_seen = np.asarray([self._last_seen[tuple(key)] for key in keys], dtype=np.int32)
        return keys, states, log_odds, observations, last_seen

    def centers(self, keys: np.ndarray) -> np.ndarray:
        return (np.asarray(keys, dtype=np.float64) + 0.5) * self.voxel_size

    def visualization_arrays(self, max_points: int = 250_000):
        """Return occupied/known centers; free voxels are transparent (omitted)."""
        keys, states, _, _, _ = self.to_arrays()
        visible = states != FREE
        keys = keys[visible]
        states = states[visible]
        if len(keys) > max_points:
            selected = np.linspace(0, len(keys) - 1, max_points, dtype=np.int64)
            keys = keys[selected]
            states = states[selected]
        points = self.centers(keys)
        colors = np.empty((len(keys), 3), dtype=np.float64)
        colors[states == KNOWN] = KNOWN_RGBA[:3]
        colors[states == OCCUPIED] = OCCUPIED_RGBA[:3]
        return points, colors, states

    def summary(self) -> Dict:
        _, states, _, _, _ = self.to_arrays()
        bounds_min = None if self._bounds_min is None else self._bounds_min.tolist()
        bounds_max = None if self._bounds_max is None else self._bounds_max.tolist()
        return {
            "voxel_size_m": self.voxel_size,
            "observed_voxels": int(len(states)),
            "free_voxels": int(np.count_nonzero(states == FREE)),
            "known_uncertain_voxels": int(np.count_nonzero(states == KNOWN)),
            "occupied_voxels": int(np.count_nonzero(states == OCCUPIED)),
            "unknown_voxels": "implicit within bounds",
            "bounds_min_key": bounds_min,
            "bounds_max_key": bounds_max,
        }

    def state_arrays(self, prefix: str = "occupancy3d") -> Dict[str, np.ndarray]:
        keys, states, log_odds, observations, last_seen = self.to_arrays()
        return {
            f"{prefix}_voxel_size": np.asarray(self.voxel_size, dtype=np.float64),
            f"{prefix}_keys": keys,
            f"{prefix}_states": states,
            f"{prefix}_log_odds": log_odds,
            f"{prefix}_observations": observations,
            f"{prefix}_last_seen": last_seen,
            f"{prefix}_bounds_min_key": np.asarray(
                self._bounds_min if self._bounds_min is not None else [0, 0, 0],
                dtype=np.int64,
            ),
            f"{prefix}_bounds_max_key": np.asarray(
                self._bounds_max if self._bounds_max is not None else [-1, -1, -1],
                dtype=np.int64,
            ),
            f"{prefix}_palette_values": np.asarray([FREE, KNOWN, OCCUPIED], dtype=np.int16),
            f"{prefix}_palette_rgba": np.vstack([FREE_RGBA, KNOWN_RGBA, OCCUPIED_RGBA]),
        }
