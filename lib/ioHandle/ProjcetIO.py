
import glob
import json
import os

import cv2
import numpy as np
from ioHandle.IOHandler import IOHandler, load_pcd
from kinematics.pose import ExtMat, Point6, Pose

class ProjcetIO(IOHandler):
    def __init__(self):
        super().__init__()

    def load_row(self, row):
        if (self.cfg["kinematics"] == "CAM2NWU"):
            extmat_path = os.path.join(self.dataset_dir, "data", "pose")
            extmat_path = os.path.join(extmat_path, row["name"] + ".json")
            with open(extmat_path, "r") as f:
                extmat_d = json.load(f)
            # extmat_data = np.loadtxt(extmat_path, delimiter=',', dtype=np.float32)
            extmat_r = extmat_d["rotation"]#.reshape(4, 4)
            extmat_t = np.array(extmat_d["translation"])#.reshape(4, 4)
            extmat_data = np.hstack([extmat_r, extmat_t.reshape(3,1)])
            extmat_data = np.vstack([extmat_data, np.array([0,0,0,1]).reshape(1,4)])
            extmat = ExtMat(data=extmat_data)
            p = Pose(extmat=extmat)
        
        elif (self.cfg["kinematics"] =="pose"):
            p6 = Point6(x=row["x"], y=row["y"], z=row["z"], 
                        roll=row["phi"], pitch=row["theta"], yaw=row["psi"])
            p = Pose(p6=p6)
        
        target_name = row["name"]
        pattern = os.path.join(self.source_dir, f"{target_name}.*")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No file found for name '{target_name}' in {self.source_dir}")

        src_path = matches[0]
        src_ext = os.path.splitext(src_path)[1]

        if src_ext == ".npy":
            src = np.load(src_path).astype(np.float32)
        elif src_ext == ".csv":
            src = np.loadtxt(src_path, delimiter=',', dtype=np.float32)
        else:
            raise ValueError("Invalid filename extension")
        
        color_path = os.path.join(self.dataset_dir, "data",
                                  "rgb", row["name"] + self.cfg["rgb_extension"])
        color_img = cv2.imread(color_path)

        dic = {
            "pose": p, 
            "source": src, 
            "image": color_img,
            "idx": row['index'],
            "name": row["name"]
        }
        return dic