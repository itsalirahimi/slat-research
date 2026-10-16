from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FeatureSet:
    points_xy: np.ndarray
    descriptors: Optional[np.ndarray]
    image_shape: Tuple[int, int]
    points_3d: Optional[np.ndarray] = None


@dataclass
class FrameData:
    index: int
    name: str
    position_utm: np.ndarray
    relative_position: np.ndarray
    points: np.ndarray
    colors: Optional[np.ndarray]
    image: Optional[np.ndarray]
    ground_points: Optional[np.ndarray]
    source_path: str
    image_path: Optional[str] = None
    ground_path: Optional[str] = None


@dataclass
class Keyframe:
    index: int
    name: str
    position_utm: np.ndarray
    relative_position: np.ndarray
    points: np.ndarray
    colors: Optional[np.ndarray]
    image: Optional[np.ndarray]
    image_path: Optional[str]
    ground_points: Optional[np.ndarray]
    transform: np.ndarray
    features: Optional[FeatureSet]
    confidence: float = 1.0


@dataclass
class FrameResult:
    index: int
    name: str
    accepted: bool
    status: str
    transform: np.ndarray
    prior_transform: np.ndarray
    residual_transform: np.ndarray
    confidence: float
    overlap_before: float = 0.0
    overlap_after: float = 0.0
    rmse_after: Optional[float] = None
    visual: Dict[str, Any] = field(default_factory=dict)
    ground: Dict[str, Any] = field(default_factory=dict)
    geometry: Dict[str, Any] = field(default_factory=dict)
    mesh: Dict[str, Any] = field(default_factory=dict)
    occupancy: Dict[str, Any] = field(default_factory=dict)
    map_2d: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    map_voxels: int = 0
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return _jsonable(self.__dict__)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
