from dataclasses import dataclass
import os
from pathlib import Path

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

@dataclass
class BGPatternDiffuserConfig:
    hfov_deg: float 
    root_dir: str
    dst_dir: str
    coarsetune_iters: int = 400
    finetune_iters: int = 800
    viz: bool = False
    coarsetune_grid_w: int = 3
    coarsetune_grid_h: int = 3
    shift_k: float = 1.2
    finetune_grid_w: int = 8
    finetune_grid_h: int = 8
    tunning_alpha: float = 1e-4
    fast: bool = False
    scoring_downsample_frac: float = 0.1
    verbosity: str = "tiny"
    spline_mesh_samples_u: int = 40
    spline_mesh_samples_v: int = 40
    spline_mesh_marginal_ratio: float = 0.05
    scoring_smoothness_k: int = 10
    scoring_smoothness_kmin_neighbors: int = 8
    scoring_smoothness_neighbors_cap: int = 64
    tunning_eps: float = 1e-3
    tunning_avgChangeTol: float = 2e-5
    tunning_varThresh: float = 100
    tunning_window: int = 10
    ct1iters: int = 200

    def __post_init__(self):
        self.diffusion_dir = os.path.join(self.root_dir, "diffusion", self.dst_dir, "background")
        self.mask_dir = os.path.join(self.root_dir, "diffusion", self.dst_dir, "mask")
        ensure_dir(self.diffusion_dir)
        ensure_dir(self.mask_dir)
        # Enforce valid verbosity value
        allowed_verbosity = ["tiny", "none", "full"]
        if self.verbosity not in allowed_verbosity:
            raise ValueError(f"Invalid verbosity value: {self.verbosity}. Allowed values are {allowed_verbosity}.")

