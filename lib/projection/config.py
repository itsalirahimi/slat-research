
from enum import Enum, auto
from dataclasses import dataclass
import os
from pathlib import Path

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

class Scaling(Enum):
    NULL = 0
    MIN_Z   = 1 
    MEAN_Z    = 2 
    RESHAPE_BG_Z  = 3 

class VisMode(Enum):
    Null = auto()
    MSingle = auto()
    MAccum = auto()

@dataclass
class Mapper3DConfig:
    hfov_deg: float
    root_dir: str
    dst_dir: str
    visMode: VisMode = VisMode.MSingle
    shape: tuple = None
    color_mode: str = 'constant'  # 'image' | 'proximity' | 'constant' | 'none'
    mesh_u: int = 40
    mesh_v: int = 40
    scaling: Scaling = Scaling.NULL
    do_save: bool = None
    on_video: bool = False
    in_camera: bool = False
    downsample_pts: int = 100000

    def __post_init__(self):
        self.radial_dir = os.path.join(self.root_dir, "projection", self.dst_dir, "radial")
        self.rgb_dir = os.path.join(self.root_dir, "projection", self.dst_dir, "rgb")
        self.canonical_dir = os.path.join(self.root_dir, "projection", self.dst_dir, "canonical")
        ensure_dir(self.radial_dir)
        ensure_dir(self.canonical_dir)
        ensure_dir(self.rgb_dir)
        if self.on_video:
            self.visMode = VisMode.MAccum

        assert (int(self.on_video) + int(self.in_camera)) <= 1, \
            "Not possible to project in camera frame (i.e. no rotation) while on video (i.e. with translation)"
