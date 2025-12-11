import os
from pathlib import Path
from ioHandle.IOHandler import IOHandler, load_pcd

class EvalIO(IOHandler):
    def __init__(self):
        super().__init__()
        self.source_dir = self.getSrcDir()
        self.dataset_dir = self.source_dir.expanduser().resolve().parent.parent.parent
        self.dst = os.path.basename(os.path.dirname(self.source_dir))
        self.init_dataset(self.dataset_dir)

    def load_row(self, row):
        
        idx = row['index']
        name = row["name"]
        # pose = self.get_pose(row)
        raw_dir = os.path.join(self.dataset_dir, "projection", self.dst, "canonical")
        raw = load_pcd(os.path.join(raw_dir, f"{name}.pcd"))
        fused_dir = os.path.join(self.dataset_dir, "fusion", self.dst, "canonical")
        fused = load_pcd(os.path.join(fused_dir, f"{name}.pcd"))
        gt_dir = os.path.join(self.dataset_dir, "eval", "gt", "pcd_ds")
        groundtruth = load_pcd(os.path.join(gt_dir, f"{name}.pcd"))

        dic = {
            "idx": idx,
            "name": name,
            "raw": raw,
            "fused": fused,
            "groundtruth": groundtruth,
        }
        return dic