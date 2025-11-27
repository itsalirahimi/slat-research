import glob
import os
import sys
import argparse
import cv2
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Union
from geom.types import Background
import open3d as o3d
from kinematics.pose import OPK, ExtMat, Point6, Pose, RotFormat
from utils.conversion import pcm2pcd

class IOHandler:
    def __init__(self):
        if len(sys.argv) < 2:
            print("Usage: python script.py --src <dataset.csv> [--index N]")
            sys.exit(1)

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--src", required=True, type=Path,
            help="Path to the source directory (e.g., ../../data/e)")
        self.parser.add_argument("--dst", type=str,
            help="Path to the output directory (e.g., ../../data/e)")

        self.parser.add_argument("--index", type=int, 
            help="Row index to use from dataset (matches 'index' column)")
        self.parser.add_argument("--start", type=int, 
            help="Row index to start from")
        
        self.parser.add_argument("--save", action="store_true", 
            help="To save the output in a corresponding dir name")
        self.parser.add_argument("--on-video", action="store_true", 
            help="Perform position translation on each projected point cloud")

        self.args = self.parser.parse_args()
        
    def getDstDir(self):
        if self.args.dst:
            return self.args.dst
        else:
            return self.dst
    
    def getDoSave(self):
        return self.args.save
    
    def getSrcDir(self):
        return Path(self.args.src)
    
    def getOnVideo(self):
        return self.args.on_video
    
    def getDataRootDir(self):
        return self.dataset_dir
    
    def init_dataset(self, dataset_dir):
        if not dataset_dir.is_dir():
            print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
            sys.exit(2)
        
        try:
            cfg_path = dataset_dir / "config.yaml"
            with open(cfg_path, 'r') as file:
                self.cfg = yaml.safe_load(file)
        except FileNotFoundError:
            cfg_path, "not found"

        data_path = os.path.join(self.dataset_dir, "data.json")
        self.df = pd.read_json(data_path)
        self.rowslist = self.df.to_dict("records")
    
    def get_rot_format(self):
        if self.cfg["kinematics"] =="pose":
            return RotFormat.AIRPLANE_EULER
        elif self.cfg["kinematics"] == "CAM2NWU":
            return RotFormat.W2C_ROT
        elif self.cfg["kinematics"] =="opk":
            return RotFormat.OPK
        else:
            raise TypeError("Invalid kinematics type")
    
    def get_color_image(self, dir, name, extention):
        color_path = os.path.join(dir, name + extention)
        return cv2.imread(color_path)
    
    def get_depth(self, target_name, dir):
        pattern = os.path.join(dir, f"{target_name}.*")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No file found for name '{target_name}' in {self.source_dir}")

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
        
        
    
    def get_pose(self, row):
        rot_format = self.get_rot_format()
        if rot_format == RotFormat.AIRPLANE_EULER:
            p6 = Point6(x=row["x"], y=row["y"], z=row["z"], 
                        roll=row["phi"], pitch=row["theta"], yaw=row["psi"])
            pose = Pose(p6, rot_format=rot_format)

        elif rot_format == RotFormat.W2C_ROT:
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
            pose = Pose(extmat, rot_format=rot_format)

        elif rot_format == RotFormat.OPK:
            # opk = OPK(x=row["pose_utm"][0], y=row["pose_utm"][1], z=row["pose_utm"][2], 
            opk = OPK(x=row["pose_utm"][0], y=row["pose_utm"][1], z=row["agl"], 
                        omega=np.deg2rad(row["OPK_deg"][0]), 
                        phi=np.deg2rad(row["OPK_deg"][1]), 
                        kappa=np.deg2rad(row["OPK_deg"][2]))
            pose = Pose(opk, rot_format=RotFormat.OPK)
        
        return pose

    def load_row(self, row):
        raise NotImplementedError("Subclasses must implement load_row()")

    def load(self):
        if self.args.index:
            row = self.df.loc[self.df["index"] == self.args.index].iloc[0]
            dic = self.load_row(row) 
            yield dic

        else:
            if self.args.start:
                matches = self.df.index[self.df["index"] == self.args.start].tolist()
                assert len(matches) > 0, f"Value {self.args.start} not found in 'index' column"
                k = matches[0]
            else:
                k = 0
            while k < len(self.rowslist):
                row = self.rowslist[k]
                dic = self.load_row(row) 
                k += 1
                yield dic

def save_pcd(points: np.ndarray, colors: np.ndarray, filepath: str):
    """
    Save a point cloud to a .pcd file.

    Args:
        points (np.ndarray): Nx3 array of 3D point coordinates.
        colors (np.ndarray): Nx3 array of RGB colors (values in [0,1]).
        filepath (str): Path to save the .pcd file.
    """
    if points.shape[1] != 3:
        raise ValueError("Points array must be of shape (N, 3)")
    if colors.shape[1] != 3:
        raise ValueError("Colors array must be of shape (N, 3)")
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Points and colors must have the same number of rows")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(float))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(float))

    # Write to file
    success = o3d.io.write_point_cloud(filepath, pcd)
    if not success:
        raise IOError(f"Failed to write point cloud to {filepath}")

def load_pcd(filepath: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a .pcd file.

    Args:
        filepath (str): Path to the .pcd file.

    Returns:
        o3d.geometry.PointCloud: Loaded point cloud.
    """
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise IOError(f"Failed to load point cloud or file is empty: {filepath}")
    return pcd

def save_pcm_as_pcd(pcm, out_path, color=None, vizimg=None):
    if vizimg is None:
        color_mat = np.zeros_like(pcm)
        if color is not None:
            color = np.asarray(color)
            color_mat[:,:,0] = color[0]
            color_mat[:,:,1] = color[1]
            color_mat[:,:,2] = color[2]
    else:
        color_mat = vizimg
    
    pcd = pcm2pcd(pcm, visualization_image=color_mat)
    save_pcd(np.asarray(pcd.points), np.asarray(pcd.colors), filepath=out_path)