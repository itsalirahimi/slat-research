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

    # Derived 2.5D mesh (PLY).
    save_mesh: bool = True
    # Fast by default: cumulative PCD checkpoints are always saved, while the
    # more expensive mesh/occupancy/RGB products are generated once at the end.
    # Enable this only when every intermediate derived product is required.
    save_derived_steps: bool = False
    mesh_grid_size: Optional[float] = None
    mesh_max_height_jump: float = 3.0
    mesh_max_edge_factor: float = 2.5

    # Sparse probabilistic 3D occupancy.  Never-observed voxels are implicit;
    # observed voxels are free, known/uncertain, or occupied.
    save_occupancy: bool = True
    occupancy_voxel_size: float = 1.0
    occupancy_resolution: Optional[float] = None  # Legacy config alias.
    occupancy_free_log_odds: float = -0.40
    occupancy_occupied_log_odds: float = 0.85
    occupancy_free_threshold: float = -0.20
    occupancy_occupied_threshold: float = 0.40
    occupancy_clamp_min: float = -2.0
    occupancy_clamp_max: float = 3.5
    occupancy_max_rays_per_frame: int = 5_000
    occupancy_max_ray_length: float = 250.0
    occupancy_ray_step_factor: float = 1.0
    occupancy_visual_max_points: int = 250_000

    # Orthographic RGB/mask and diagnostic products.
    save_map_2d: bool = True
    map_2d_resolution: Optional[float] = None
    map_2d_padding: float = 1.0
    map_2d_max_pixels: int = 16_000_000
    save_diagnostics: bool = True
    diagnostic_max_matches: int = 250
    diagnostic_max_image_dimension: int = 1_600
    diagnostic_max_geometry_points: int = 100_000

    # Optional second-pass loop closure.  Incremental products are never
    # overwritten; optimized products are written below output/optimized/.
    run_loop_closure: bool = False
    loop_min_separation: int = 3
    loop_search_radius: float = 25.0
    loop_max_candidates: int = 20
    loop_min_overlap: float = 0.12
    loop_min_inliers: int = 15
    loop_max_rmse: Optional[float] = None
    pose_graph_translation_sigma: float = 1.0
    pose_graph_rotation_sigma_deg: float = 3.0
    pose_graph_robust_scale: float = 1.0
    pose_graph_max_iterations: int = 100

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
        if self.mesh_grid_size is None:
            self.mesh_grid_size = self.voxel_size
        if self.occupancy_resolution is not None:
            # Preserve configurations written for the first mapping draft.
            self.occupancy_voxel_size = float(self.occupancy_resolution)
        if self.map_2d_resolution is None:
            self.map_2d_resolution = self.voxel_size
        if self.loop_max_rmse is None:
            self.loop_max_rmse = 2.0 * self.voxel_size

        if self.overlap_distance <= 0 or self.max_icp_rmse <= 0:
            raise ValueError("mapping distance thresholds must be positive")
        if self.mesh_grid_size <= 0 or self.mesh_max_height_jump <= 0:
            raise ValueError("mesh grid size and height jump must be positive")
        if self.occupancy_voxel_size <= 0:
            raise ValueError("occupancy_voxel_size must be positive")
        if not 0 < self.occupancy_ray_step_factor <= 1:
            raise ValueError("occupancy_ray_step_factor must be in (0, 1]")
        if self.occupancy_max_rays_per_frame < 1:
            raise ValueError("occupancy_max_rays_per_frame must be positive")
        if self.map_2d_resolution <= 0 or self.map_2d_max_pixels < 1:
            raise ValueError("2D map resolution and max pixels must be positive")
        if self.loop_min_separation < 2 or self.loop_max_candidates < 1:
            raise ValueError("loop closure separation/candidate limits are invalid")
        if self.loop_search_radius <= 0 or self.loop_max_rmse <= 0:
            raise ValueError("loop closure distance thresholds must be positive")

        self.output_dir = self.root_dir / "mapping" / self.dst_dir
        self.steps_dir = self.output_dir / "steps"
        self.reports_dir = self.output_dir / "reports"
        self.mesh_steps_dir = self.output_dir / "mesh_steps"
        self.occupancy_dir = self.output_dir / "occupancy"
        self.occupancy_steps_dir = self.occupancy_dir / "steps"
        self.map_2d_dir = self.output_dir / "map_2d"
        self.map_2d_steps_dir = self.map_2d_dir / "steps"
        self.visual_reports_dir = self.reports_dir / "visual"
        self.geometry_reports_dir = self.reports_dir / "geometry"
        self.optimized_dir = self.output_dir / "optimized"

        if self.do_save:
            self.steps_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            if self.save_derived_steps and self.save_mesh:
                self.mesh_steps_dir.mkdir(parents=True, exist_ok=True)
            if self.save_occupancy:
                self.occupancy_dir.mkdir(parents=True, exist_ok=True)
            if self.save_derived_steps and self.save_occupancy:
                self.occupancy_steps_dir.mkdir(parents=True, exist_ok=True)
            if self.save_map_2d:
                self.map_2d_dir.mkdir(parents=True, exist_ok=True)
            if self.save_derived_steps and self.save_map_2d:
                self.map_2d_steps_dir.mkdir(parents=True, exist_ok=True)
            if self.save_diagnostics:
                self.visual_reports_dir.mkdir(parents=True, exist_ok=True)
                self.geometry_reports_dir.mkdir(parents=True, exist_ok=True)
            if self.run_loop_closure:
                self.optimized_dir.mkdir(parents=True, exist_ok=True)
