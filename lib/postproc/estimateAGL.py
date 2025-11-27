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
import open3d as o3d

from ioHandle.IOHandler import save_pcd
from kinematics.pose import OPK, ExtMat, Point6, Pose, RotFormat
from projection.helper import project3D
from utils.conversion import pcm2pcd

def find_centroid(pcd_path: str) -> np.ndarray:
    """
    Load a PCD file, compute its centroid (CG), print it, and return it.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=float)

    if pts.size == 0:
        raise ValueError(f"Point cloud is empty: {pcd_path}")

    # Compute centroid
    centroid = pts.mean(axis=0)

    return centroid

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

def main():
    ap = argparse.ArgumentParser(description="lidar projection")
    ap.add_argument("--src", required=True, help="lidar data path")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)

    dataset_dir = Path(os.path.join(src_dir, "../../../.."))
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
        pcd_path = os.path.join(src_dir, f"{name}.pcd")

        try:
            centroid = find_centroid(pcd_path)
            print(f"z_cg = {centroid[2]:.6f}, z_cam = {pose.p6.z}")
            df.loc[k, "agl"] = abs(float(centroid[2]))
            k += 1
        except:
            break


    df.to_json(data_path, orient="records", indent=2)






if __name__ == "__main__":
    main()
