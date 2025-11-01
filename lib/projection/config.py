
from enum import Enum, auto
from dataclasses import dataclass
import os

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
    pyramidProj: bool = False

    def __post_init__(self):
        self.output_dir = os.path.join(self.root_dir, "projection", self.dst_dir)
        if self.on_video:
            self.visMode = VisMode.MAccum

        assert (int(self.on_video) + int(self.in_camera)) <= 1, \
            "Not possible to project in camera frame (i.e. no rotation) while on video (i.e. with translation)"
