import os
import sys

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from utils.conversion import pcd2pcm
from ioHandle.IOHandler import load_pcd
from ioHandle.ProjcetIO import ProjcetIO

io = ProjcetIO()

pre_path      = "data/usegeo_1"
H, W          = 257, 388
in_dir        = [f"{pre_path}/fusion/test/ground",
                    f"{pre_path}/fusion/test/fused",
                    f"{pre_path}/diffusion/test/canonical/background",
                    f"{pre_path}/projection/test/canonical"]

for i in in_dir:
    out_dir = f"{i}_mhw3_p"
    os.makedirs(out_dir, exist_ok=True)

    for dict in io.load():
        # Generate a random point cloud and update the app
        pose = dict["pose"]
        name = dict["name"]
        image = dict["image"]

        in_file = os.path.join(i, f"{name}.pcd")
        out_file = os.path.join(out_dir, f"{name}.npy")
        pcd_nwu = load_pcd(in_file)
        pcm_nwu = pcd2pcm(pcd_nwu, H, W)
        R = pose.getCAM2NWU()
        pc_cam = pcm_nwu @ R

        np.save(out_file, pc_cam)

        print(f"[info] {out_file} {dict['idx']}.")