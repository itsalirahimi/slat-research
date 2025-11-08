
import glob
import os

import cv2

from ioHandle.IOHandler import IOHandler, load_pcd

class DiffuseIO(IOHandler):
    def __init__(self):
        super().__init__()
        assert self.args.dst, "Must set --dst when diffusing."
    
    def load_row(self, row):
        target_name = row["name"]
        pattern = os.path.join(self.source_dir, f"{target_name}.*")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No file found for name '{target_name}' in {self.source_dir}")

        src_path = matches[0]
        src_ext = os.path.splitext(src_path)[1]

        if src_ext == ".pcd":
            src = load_pcd(src_path)
        else:
            raise ValueError("Invalid filename extension")
        
        color_path = os.path.join(self.source_dir, "..",
                                  "rgb", row["name"] + self.cfg["rgb_extension"])
        color_img = cv2.imread(color_path)
        dic = {
            "source": src, 
            "image": color_img, 
            "idx": row['index'],
            "name": row['name'],
        }
        return dic
