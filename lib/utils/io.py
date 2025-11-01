import os
import sys
import argparse
import yaml
from pathlib import Path
import pandas as pd
from kinematics.pose import ExtMat, Point6, Pose
import cv2
import numpy as np
import json
from typing import Union
from geom.types import Background
import open3d as o3d
from utils.conversion import pcm2pcd
import glob

def _find_config(dataset_dir: Path) -> Path:
    """Return the path to the first existing YAML config file in dataset_dir."""
    for name in ("config.yaml", "config.yml"):
        candidate = dataset_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No config.yaml or config.yml found in {dataset_dir}")

class IOHandler:
    def __init__(self):
        if len(sys.argv) < 2:
            print("Usage: python script.py --dataset <dataset.csv> [--index N]")
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
        self.parser.add_argument("--in-camera", action="store_true", 
            help="Perform projection in all of the subroutines, in camera frame. ATTENTION: YOU MUST SET THIS FLAG BOTH IN FUSION AND DIFFUSION OTHERWISE CORRUPTED OUTPUTS!")

        self.args = self.parser.parse_args()


        # Resolve the dataset path relative to the current working directory
        self.src = self.args.src.name
        if self.args.dst:
            self.dst = self.args.dst
        else:
            self.dst = self.src
        
        self.dataset_dir = Path(self.args.src).expanduser().resolve().parent.parent
        if not self.dataset_dir.is_dir():
            print(f"Dataset directory not found: {self.dataset_dir}", file=sys.stderr)
            sys.exit(2)

        # Attempt to load the config.yaml file
        try:
            cfg_path = _find_config(self.dataset_dir)
            with open(cfg_path, 'r') as file:
                self.cfg = yaml.safe_load(file)
        except FileNotFoundError:
            cfg_path, "not found"

        self.df = pd.read_csv(os.path.join(self.dataset_dir, "data.csv"))
        self.rowslist = self.df.to_dict("records")

        self.source_dir = self.args.src
        self.bgDir = os.path.join(self.dataset_dir, "diffusion", self.dst)

    def getDstDir(self):
        return self.dst
    
    def getDoSave(self):
        return self.args.save
    
    def getOnVideo(self):
        return self.args.on_video
    
    def getDataRootDir(self):
        return self.dataset_dir
    
    def getInCamera(self):
        return self.args.in_camera
    
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

class FuseIO(IOHandler):
    def __init__(self):
        super().__init__()

    def load_row(self, row):
        if (self.cfg["kinematics"] == "CAM2NWU"):
            extmat_path = os.path.join(self.dataset_dir, "extmat")
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
        elif src_ext == ".pcd":
            src = load_pcd(src_path)
        else:
            raise ValueError("Invalid filename extension")
        
        color_path = os.path.join(self.dataset_dir, 
                                  "rgb", row["name"] + self.cfg["rgb_extension"])
        color_img = cv2.imread(color_path)
        bg = load_pcd(os.path.join(self.bgDir, f"{row['name']}.pcd"))
        dic = {
            "pose": p, 
            "source": src, 
            "image": color_img, 
            "bg": bg,
            "idx": row['index'],
            "name": row["name"]
        }
        return dic

class EvalIO(IOHandler):
    def __init__(self):
        super().__init__()

class DiffuseIO(IOHandler):
    def __init__(self):
        super().__init__()
    
    def load_row(self, row):
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
        elif src_ext == ".pcd":
            src = load_pcd(src_path)
        else:
            raise ValueError("Invalid filename extension")
        
        color_path = os.path.join(self.dataset_dir, 
                                  "rgb", row["name"] + self.cfg["rgb_extension"])
        color_img = cv2.imread(color_path)
        
        dic = {
            "source": src, 
            "image": color_img, 
            "idx": row['index'],
            "name": row['name'],
        }
        return dic

class ProjcetIO(IOHandler):
    def __init__(self):
        super().__init__()

    def load_row(self, row):
        if (self.cfg["kinematics"] == "CAM2NWU"):
            extmat_path = os.path.join(self.dataset_dir, "extmat")
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
        elif src_ext == ".pcd":
            src = load_pcd(src_path)
        else:
            raise ValueError("Invalid filename extension")
        
        color_path = os.path.join(self.dataset_dir, 
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


ArrayLike = Union[np.ndarray, list, tuple]

def save_grid_with_Tinv(
    json_path: str,
    points_xyz: ArrayLike,   # (N,3)
    grid_w: int,
    grid_h: int,
    T: ArrayLike,            # (4,4) forward homogeneous transform
    float_precision: int = 6
) -> None:
    """
    Save grid points + metadata + T_inv into a JSON file.

    Parameters
    ----------
    json_path : str
        Where to write the JSON file.
    points_xyz : (N,3) array-like
        Grid points (rows = [x,y,z]).
    grid_w, grid_h : float
        Grid width and height.
    T : (4,4) array-like
        Forward homogeneous transform (Original -> Oriented).
    float_precision : int
        Decimal places to round when saving.
    """
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {pts.shape}")

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4), got {T.shape}")

    # Compute inverse analytically
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    # Round for nicer JSON
    pts_out = np.round(pts, float_precision).tolist()
    T_inv_out = np.round(T_inv, float_precision).tolist()

    payload = {
        "grid": {
            "width": int(grid_w),
            "height": int(grid_h),
            "units": "meters"
        },
        "points_xyz": pts_out,
        "T_inv": T_inv_out,
        "metadata": {
            "matrix_format": "row-major",
            "convention": "x_restored = T_inv * x_oriented",
            "description": "Grid points with inverse transform."
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_background(json_path: str) -> Background:
    """
    Load a Background instance from a JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    Background
        Instance containing grid points, dimensions, T_inv, metadata.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points_xyz = np.array(data["points_xyz"], dtype=float)
    grid_w = int(data["grid"]["width"])
    grid_h = int(data["grid"]["height"])
    T_inv = np.array(data["T_inv"], dtype=float)
    metadata = data.get("metadata", {})

    return Background(
        points_xyz=points_xyz,
        grid_w=grid_w,
        grid_h=grid_h,
        T_inv=T_inv,
        metadata=metadata
    )

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