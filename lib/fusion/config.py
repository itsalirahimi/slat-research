from enum import Enum, auto
from dataclasses import dataclass
import os
from pathlib import Path

from projection.config import VisMode

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

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
    visMode: VisMode = VisMode.MSingle
    do_save: bool = None
    on_video: bool = False


    def __post_init__(self):
        self.fusion_dir = os.path.join(self.root_dir, "fusion", self.dst_dir, "fused")
        self.gep_dir = os.path.join(self.root_dir, "fusion", self.dst_dir, "ground")
        ensure_dir(self.fusion_dir)
        ensure_dir(self.gep_dir)
