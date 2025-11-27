import os
from pathlib import Path
from ioHandle.IOHandler import IOHandler

class ProjcetIO(IOHandler):
    def __init__(self):
        super().__init__()
        self.source_dir = self.getSrcDir()
        self.dataset_dir = self.source_dir.expanduser().resolve().parent.parent
        self.dst = os.path.basename(os.path.dirname(self.source_dir))
        self.init_dataset(self.dataset_dir)

    def load_row(self, row):
        
        idx = row['index']
        name = row["name"]
        pose = self.get_pose(row)
        depth = self.get_depth(name, self.source_dir)
        img_dir = os.path.join(self.dataset_dir, "rgb")
        color_img = self.get_color_image(img_dir , name, self.cfg["rgb_extension"])

        dic = {
            "idx": idx,
            "name": name,
            "pose": pose, 
            "source": depth, 
            "image": color_img,
        }
        return dic