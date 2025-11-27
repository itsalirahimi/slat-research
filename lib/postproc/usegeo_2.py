#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path
import shutil
from typing import List, Dict, Any
import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")

import yaml


from ioHandle.IOHandler import save_pcd
from kinematics.pose import OPK, ExtMat, Point6, Pose, RotFormat
from projection.helper import project3D
from utils.conversion import pcm2pcd

def get_pose(row, kinematics):
    if kinematics =="pose":
            rot_format = RotFormat.AIRPLANE_EULER
    elif kinematics == "CAM2NWU":
        rot_format = RotFormat.W2C_ROT
    elif kinematics =="opk":
        rot_format = RotFormat.OPK
    else:
        raise TypeError("Invalid kinematics type")
    
    if rot_format == RotFormat.AIRPLANE_EULER:
        p6 = Point6(x=row["x"], y=row["y"], z=row["z"], 
                    roll=row["phi"], pitch=row["theta"], yaw=row["psi"])
        pose = Pose(p6, rot_format=rot_format)

    elif rot_format == RotFormat.W2C_ROT:
        # extmat_path = os.path.join(dataset_dir, "data", "pose")
        # extmat_path = os.path.join(extmat_path, row["name"] + ".json")
        # with open(extmat_path, "r") as f:
        #     extmat_d = json.load(f)
        # # extmat_data = np.loadtxt(extmat_path, delimiter=',', dtype=np.float32)
        # extmat_r = extmat_d["rotation"]#.reshape(4, 4)
        # extmat_t = np.array(extmat_d["translation"])#.reshape(4, 4)
        # extmat_data = np.hstack([extmat_r, extmat_t.reshape(3,1)])
        # extmat_data = np.vstack([extmat_data, np.array([0,0,0,1]).reshape(1,4)])
        # extmat = ExtMat(data=extmat_data)
        # pose = Pose(extmat, rot_format=rot_format)
        pass

    elif rot_format == RotFormat.OPK:
        opk = OPK(x=row["pose_utm"][0], y=row["pose_utm"][1], z=row["agl"], 
                  omega=np.deg2rad(row["OPK_deg"][0]), 
                  phi=np.deg2rad(row["OPK_deg"][1]), 
                  kappa=np.deg2rad(row["OPK_deg"][2]))
        pose = Pose(opk, rot_format=RotFormat.OPK)
    
    return pose

def get_depth(target_name, dir):
    pattern = os.path.join(dir, f"{target_name}.*")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No file found for name '{target_name}' in {dir}")

    src_path = matches[0]
    src_ext = os.path.splitext(src_path)[1]

    if src_ext == ".npy":
        return np.load(src_path).astype(np.float32)

    elif src_ext == ".csv":
        return np.loadtxt(src_path, delimiter=',', dtype=np.float64)

    elif src_ext == ".npz":
        data = np.load(src_path)
        if "depth" not in data:
            raise ValueError(f"NPZ file {src_path} does not contain 'depth'")
        return data["depth"].astype(np.float32)

    else:
        raise ValueError("Invalid filename extension")
            
def lidar_projection(cimg, depth, pose, hfov_deg):
    projected_depth, _ = project3D(depth, pose, hfov_deg, 
        move=False, pyramidProj=False, do_rotate=True, do_scale=False)

    return pcm2pcd(projected_depth, cimg)

def save_pcd_file(pcd, name, out_dir):
    _filepath = os.path.join(out_dir, f"{name}.pcd")
    save_pcd(np.asarray(pcd.points), np.asarray(pcd.colors), filepath=_filepath)



def main():
    ap = argparse.ArgumentParser(description="lidar projection")
    ap.add_argument("--src", required=True, help="lidar data path")
    ap.add_argument("--out", default=None, help="Output directory")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)
    img_dir = os.path.join(src_dir, "../rgb")
    out_dir = os.path.join(src_dir, "../pcd_ds")
    os.makedirs(out_dir, exist_ok=True)

    dataset_dir = Path(os.path.join(src_dir, "../../.."))
    if not dataset_dir.is_dir():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(2)
    
    try:
        cfg_path = dataset_dir / "config.yaml"
        with open(cfg_path, 'r') as file:
            cfg = yaml.safe_load(file)
    except FileNotFoundError:
        cfg_path, "not found"

    data_path = os.path.join(dataset_dir, "data.json")
    df = pd.read_json(data_path)
    rowslist = df.to_dict("records")
    
    k = 0
    c_row = len(rowslist)
    while k < c_row:
        
        row = rowslist[k]
        name = row["name"]
        kinematics = cfg["kinematics"]
        pose = get_pose(row, kinematics)
        color_path = os.path.join(img_dir, name + cfg["rgb_extension"])
        color_img = cv2.imread(color_path)
        depth = get_depth(name, src_dir)
        pcd_out = lidar_projection(color_img, depth, pose, cfg["hfov_deg"])
        save_pcd_file(pcd_out, name, out_dir)
        k += 1
        print(f"[{k}/{c_row}] OK {name}")
    






if __name__ == "__main__":
    main()
