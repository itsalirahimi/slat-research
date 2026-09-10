from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MappingConfig:
    """Runtime options for incremental mapping.

    Defaults are intentionally conservative for the approximately 80 m AGL
    aerial sequence supplied with this project.  Every threshold is in meters
    unless its name states otherwise.
    """

    root_dir: Path
    dst_dir: str
    do_save: bool = False
    utm_zone: Optional[str] = None
    position_mode: str = "relative"
    voxel_size: float = 0.50

    # Overlap and local-map selection.
    overlap_distance: Optional[float] = None
    min_overlap: float = 0.05
    local_map_margin: float = 10.0
    max_registration_points: int = 80_000

    # Registration acceptance.
    min_icp_fitness: float = 0.08
    max_icp_rmse: Optional[float] = None
    max_residual_translation: float = 10.0
    max_residual_rotation_deg: float = 12.0

    # Visual matching.
    use_visual: bool = True
    max_keyframe_candidates: int = 5
    sift_features: int = 8_000
    sift_max_dimension: int = 1_600
    sift_ratio: float = 0.75
    visual_ransac_px: float = 3.0
    visual_3d_threshold: Optional[float] = None
    visual_min_inliers: int = 10
    visual_min_coverage: float = 0.003
    visual_ransac_iterations: int = 1_500
    max_visual_icp_disagreement_m: float = 3.0
    max_visual_icp_disagreement_deg: float = 5.0

    # Generated-ground-plane (GEP) constraint.
    use_ground: bool = True
    ground_voxel_size: float = 1.0
    ground_max_tilt_deg: float = 4.0
    ground_max_shift: float = 2.0
    ground_min_points: int = 50

    # Voxel/color fusion.
    max_voxel_weight: float = 30.0
    color_sigma: float = 0.20
    color_reject_distance: float = 0.45
    random_seed: int = 7

    def __post_init__(self):
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.dst_dir = str(self.dst_dir)

        if self.position_mode not in {"relative", "off"}:
            raise ValueError("position_mode must be 'relative' or 'off'")
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if not 0.0 <= self.min_overlap <= 1.0:
            raise ValueError("min_overlap must be between 0 and 1")
        if self.max_registration_points < 100:
            raise ValueError("max_registration_points must be at least 100")

        if self.overlap_distance is None:
            self.overlap_distance = 3.0 * self.voxel_size
        if self.max_icp_rmse is None:
            self.max_icp_rmse = 2.5 * self.voxel_size
        if self.visual_3d_threshold is None:
            self.visual_3d_threshold = 3.0 * self.voxel_size

        if self.overlap_distance <= 0 or self.max_icp_rmse <= 0:
            raise ValueError("mapping distance thresholds must be positive")

        self.output_dir = self.root_dir / "mapping" / self.dst_dir
        self.steps_dir = self.output_dir / "steps"
        self.reports_dir = self.output_dir / "reports"

        if self.do_save:
            self.steps_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
