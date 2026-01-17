#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict, Any
import cv2
import numpy as np
import open3d as o3d

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from utils.conversion import pcd2hw1, pcd2pcm

def load_ortholoc_npz(npz_path):
    """
    Load OrthoLoC dataset npz

    Args:
        npz_path (str): Path to the npz file.

    Returns:
        image_query (np.ndarray): OpenCV image (H,W,3).
        filename (str): npz file name without extension.
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3,).
        intrinsics (np.ndarray): Intrinsic matrix (3x3).
        point_map (np.ndarray): Point map (H,W,3).
    """


    data = np.load(npz_path, allow_pickle=True)

    # Extract arrays
    image_query = data["image_query"]        # (H,W,3)
    extrinsics_ext = data["extrinsics_refined"]  # (3,4) or (4,4)
    intrinsics = data["intrinsics"]          # (3,3)
    point_map = data["point_map"]            # (H,W,3)

    H, W, _ = image_query.shape

    # print(f"image_query: {image_query.shape}")
    # print(f"extrinsics_ext: {extrinsics_ext.shape}")
    # print(f"intrinsics: {intrinsics.shape}")
    # print(f"point_map: {point_map.shape}")

    # Handle extrinsics_extended 3x4
    R = extrinsics_ext[:, :3]
    t = extrinsics_ext[:, 3]

    # Ensure image is uint8 for OpenCV
    if image_query.dtype != np.uint8:
        image_query = (255 * (image_query - image_query.min()) /
                       (image_query.max() - image_query.min())).astype(np.uint8)

    filename = os.path.splitext(os.path.basename(npz_path))[0]

    return image_query, filename, R, t, intrinsics, point_map

def save_csv(csv_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    try:
        H, W, _ = arr.shape
        arr = arr.reshape((H, W))
    except:
        pass
    np.savetxt(csv_path, arr, delimiter=",", fmt="%.6f")

def save_npz(npz_path: str, arr: np.ndarray) -> None:
    """
    Save depth array as compressed npz.
    Key name 'depth' to match depth.npy-like semantics.
    """
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, depth=arr)


def main():
    ap = argparse.ArgumentParser(description="Usegeo Batch-converter")
    ap.add_argument("--raw", required=True, help="Raw data path containing depth_maps, undistorted_images and Image_orientations_dataset.xyz")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()


    downsample_pts = 100_000
    raw_root = os.path.abspath(args.raw)
    out_root = os.path.abspath(args.out) if args.out is not None else raw_root


    # Output dirs
    csv_dir = os.path.join(out_root, "eval", "gt", "csv")
    npz_dir = os.path.join(out_root, "eval", "gt", "npz")
    pcd_dir = os.path.join(out_root, "eval", "gt", "pcd")
    rgb_dir = os.path.join(out_root, "rgb")
    csv_ds_dir = os.path.join(out_root, "eval", "gt", "csv_ds")
    npz_ds_dir = os.path.join(out_root, "eval", "gt", "npz_ds")
    rgb_ds_dir = os.path.join(out_root, "eval", "gt", "rgb")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(csv_ds_dir, exist_ok=True)
    os.makedirs(npz_ds_dir, exist_ok=True)
    os.makedirs(rgb_ds_dir, exist_ok=True)

    npz_files: List[str] = []

    for f in os.listdir(raw_root):
        p = os.path.join(raw_root, f)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (".npz"):
            npz_files.append(p)
    npz_files.sort()

    done = 0
    failed = 0

    json_rows: list[Dict[str, Any]] = []
    

    for idx, npz_path in enumerate(npz_files, 1):

        image_query, filename, R, t, intrinsics, point_map = load_ortholoc_npz(npz_path)

        # Save image
        img_path = os.path.join(rgb_dir, f"{filename}.png")
        H, W, _ = image_query.shape
        image_q = np.zeros_like(image_query)
        for i in range(H):
            for j in range(W):
                image_q[i][j] = [image_query[i][j][2],image_query[i][j][1], image_query[i][j][0]]
        cv2.imwrite(img_path, image_q)

        

        points = point_map.reshape(-1, 3)
        colors = image_query.reshape(-1, 3) / 255.0
        gt_pcm = o3d.geometry.PointCloud()
        gt_pcm.points = o3d.utility.Vector3dVector(points)
        gt_pcm.colors = o3d.utility.Vector3dVector(colors)
        # gt_depth = pcd2hw1(gt_pcm, H, W)
        _pcm = pcd2pcm(gt_pcm, H, W)
        gt_depth = np.zeros(shape=(H,W,1))
        print(gt_depth.shape)

        def dist(p1, p2):
            dx = (abs(0 - p2[0]))**2
            dy = (abs(0 - p2[1]))**2
            dz = (abs(p1[2] - p2[2]))**2
            return (dx+dy+dz)**0.5

        for i in range(H):
            for j in range(W):
                d = dist(t, _pcm[i][j])
                gt_depth[i][j][0] = d
                x, y, z = _pcm[i][j]
        
        # _norm = np.linalg.norm(_pcm, axis=2)    # (H, W)
        # gt_depth = _norm[..., np.newaxis]       # (H, W, 1)

        pcd_path = os.path.join(pcd_dir, f"{filename}.pcd")
        o3d.io.write_point_cloud(pcd_path, gt_pcm)
        print(filename)
        print(gt_depth.shape)



        csv_path = os.path.join(csv_dir, f"{filename}.csv")
        npz_path = os.path.join(npz_dir, f"{filename}.npz")
        save_csv(csv_path, gt_depth)
        save_npz(npz_path, gt_depth)

        ds_ratio = (downsample_pts / (H*W)) ** 0.5
        W_ds = int(ds_ratio*W)
        H_ds = int(ds_ratio*H)
        arr_ds = cv2.resize(gt_depth, (W_ds, H_ds), interpolation=cv2.INTER_AREA)
        cimg_ds = cv2.resize(image_q, (W_ds, H_ds), interpolation=cv2.INTER_AREA)

        csv_ds_path = os.path.join(csv_ds_dir, f"{filename}.csv")
        npz_ds_path = os.path.join(npz_ds_dir, f"{filename}.npz")
        rgb_ds_path = os.path.join(rgb_ds_dir, f"{filename}.png")
        save_csv(csv_ds_path, arr_ds)
        save_npz(npz_ds_path, arr_ds)
        cv2.imwrite(rgb_ds_path, cimg_ds)

        # Build JSON row
        row = {
            "index": len(json_rows),
            "name": filename,
            "pose_utm": t.tolist(),
            "agl": float(t[2]),
            "rotation": R.tolist(),
            "intrinsic": intrinsics.tolist(),
        }
        json_rows.append(row)
        done += 1
        print(f"[{idx}/{len(npz_files)}] OK {filename}")



    # Write data.json
    data_json_path = os.path.join(out_root, "data.json")
    try:
        with open(data_json_path, "w") as f:
            json.dump(json_rows, f, indent=2)
        print(f"\n[i] Wrote JSON with {len(json_rows)} entries to {data_json_path}")
    except Exception as e:
        print(f"[E] Failed to write data.json: {e}", file=sys.stderr)
        failed += 1

    print("\n[Summary]")
    print(f"  converted:           {done}")
    print(f"  failed:              {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
