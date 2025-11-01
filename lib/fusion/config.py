from enum import Enum, auto
from dataclasses import dataclass
import os

from projection.config import VisMode

class FlatFusionMode(Enum):
    Replace_25D = auto()
    Drop = auto()
    Unfold = auto()
    NDFDrop = auto()
    Depyramidize = auto()

@dataclass
class BGPatternFuserConfig:
    hfov_deg: float
    root_dir: str
    dst_dir: str
    flat_mode: FlatFusionMode = FlatFusionMode.NDFDrop
    downsample_dstW: int = 1000
    in_camera: bool = False
    visMode: VisMode = VisMode.MSingle
    do_save: bool = None
    on_video: bool = False

    def __post_init__(self):
        self.output_dir = os.path.join(self.root_dir, "fusion", self.dst_dir)
