import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../lib")

from ioHandle.IOHandler import load_pcd
from utils.conversion import pcd2hw1, pcd2pcm

def convert_pcd_to_npy(in_dir, out_dir, H, W, output_format):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if fname.lower().endswith(".pcd"):

            pcd_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".npy")

            pcd = load_pcd(pcd_path)

            if output_format == "MHW1":
                out = pcd2hw1(pcd, H, W)
            elif output_format == "MHW3":
                out = pcd2pcm(pcd, H, W)
            elif output_format == "NHW1":
                hw1 = pcd2hw1(pcd, H, W)
                max_val = np.max(hw1)
                out = hw1 * 1/max_val
            elif output_format == "NHW3":
                pcm = pcd2pcm(pcd, H, W)
                hw1 = pcd2hw1(pcd, H, W)
                max_val = np.max(hw1)
                out = pcm * 1/max_val
            
            np.save(out_path, out)

            print(f"Converted: {fname} -> {out_path} [{output_format}]")

    print("Done!")

pre_path      = "data/usegeo_3"
H, W          = 257, 388
# output_format: MHW1, MHW3, NHW1, NHW3

in_dir        = f"{pre_path}/eval/gt/pcd_ds"
out_dir       = f"{pre_path}/eval/gt/mhw1_ds"
output_format = "MHW1"
convert_pcd_to_npy(in_dir, out_dir, H, W, output_format)

in_dir        = f"{pre_path}/projection/test/canonical"
out_dir       = f"{pre_path}/projection/test/depthcan_mhw1"
output_format = "MHW1"
convert_pcd_to_npy(in_dir, out_dir, H, W, output_format)

in_dir        = f"{pre_path}/fusion/test/fused"
out_dir       = f"{pre_path}/fusion/test/fused_mhw1"
output_format = "MHW1"
convert_pcd_to_npy(in_dir, out_dir, H, W, output_format)

in_dir        = f"{pre_path}/fusion/test/ground"
out_dir       = f"{pre_path}/fusion/test/gep_mhw1"
output_format = "MHW1"
convert_pcd_to_npy(in_dir, out_dir, H, W, output_format)

in_dir        = f"{pre_path}/fusion/test/ground"
out_dir       = f"{pre_path}/fusion/test/gep_mhw3"
output_format = "MHW3"
convert_pcd_to_npy(in_dir, out_dir, H, W, output_format)




