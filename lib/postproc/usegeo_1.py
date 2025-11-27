#!/usr/bin/env python3
import argparse
import json
import os
import sys
import shutil
from typing import List, Dict, Any
import cv2
import numpy as np
import rasterio
from PIL import Image  # noqa: F401  (kept in case you want to validate images later)


def read_tiff_as_array(path: str) -> np.ndarray:
    """Return a 2D numpy array (H, W). If multi-band, use the first band."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)  # first band

    if arr.ndim == 3:
        # Use the first band if multi-channel
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape {arr.shape} from {path}")
    return arr.astype(np.float32, copy=False)


def save_csv(csv_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    np.savetxt(csv_path, arr, delimiter=",", fmt="%.6f")

def save_npz(npz_path: str, arr: np.ndarray) -> None:
    """
    Save depth array as compressed npz.
    Key name 'depth' to match depth.npy-like semantics.
    """
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, depth=arr)

def is_tiff(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".tif", ".tiff")


def collect_tiffs(root: str, recursive: bool) -> List[str]:
    if os.path.isfile(root):
        return [root] if is_tiff(root) else []

    files: List[str] = []
    if recursive:
        for d, _, fnames in os.walk(root):
            for f in fnames:
                p = os.path.join(d, f)
                if is_tiff(p):
                    files.append(p)
    else:
        if not os.path.isdir(root):
            return []
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and is_tiff(p):
                files.append(p)
    files.sort()
    return files


def parse_orientations(xyz_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse Image_orientations_dataset1.xyz.

    Returns:
        dict:
          stem -> {
            "pose_utm": [X0, Y0, Z0],
            "OPK_deg": [omega, phi, kappa],
            "intrinsic": [[c, 0, 0],
                          [0, c, 0],
                          [x0, y0, 1]]
          }
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    if not os.path.isfile(xyz_path):
        print(f"[W] Orientation file not found: {xyz_path}", file=sys.stderr)
        return mapping

    with open(xyz_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect at least: label X0 Y0 Z0 omega phi kappa c x0 y0 ...
            if len(parts) < 10:
                print(f"[W] Skipping malformed line in {xyz_path}: {line}", file=sys.stderr)
                continue

            label = parts[0]
            stem = os.path.splitext(os.path.basename(label))[0]

            try:
                X0 = float(parts[1])
                Y0 = float(parts[2])
                Z0 = float(parts[3])
                omega = float(parts[4])
                phi = float(parts[5])
                kappa = float(parts[6])
                c = float(parts[7])
                x0 = float(parts[8])
                y0 = float(parts[9])
            except ValueError:
                print(f"[W] Skipping line with non-numeric values: {line}", file=sys.stderr)
                continue

            pose_utm = [X0, Y0, Z0]
            OPK_deg = [omega, phi, kappa]
            intrinsic = [
                [c, 0.0, 0.0],
                [0.0, c, 0.0],
                [x0, y0, 1.0],
            ]

            mapping[stem] = {
                "pose_utm": pose_utm,
                "OPK_deg": OPK_deg,
                "intrinsic": intrinsic,
            }

    return mapping


def find_image_for_stem(images_dir: str, stem: str) -> str | None:
    """Try to find an image file with given stem in images_dir."""
    if not os.path.isdir(images_dir):
        return None

    exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    for ext in exts:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description="Usegeo Batch-converter")
    ap.add_argument("--raw", required=True, help="Raw data path containing depth_maps, undistorted_images and Image_orientations_dataset1.xyz")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    downsample_pts = 100_000

    raw_root = os.path.abspath(args.raw)
    out_root = os.path.abspath(args.out) if args.out is not None else raw_root

    depth_dir = os.path.join(raw_root, "depth_maps")
    images_dir = os.path.join(raw_root, "undistorted_images")
    orient_path = os.path.join(raw_root, "Image_orientations_dataset1.xyz")

    # Output dirs
    csv_dir = os.path.join(out_root, "eval", "gt", "csv")
    npz_dir = os.path.join(out_root, "eval", "gt", "npz")
    rgb_dir = os.path.join(out_root, "rgb")

    csv_ds_dir = os.path.join(out_root, "eval", "gt", "csv_ds")
    npz_ds_dir = os.path.join(out_root, "eval", "gt", "npz_ds")
    rgb_ds_dir = os.path.join(out_root, "eval", "gt", "rgb")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    os.makedirs(csv_ds_dir, exist_ok=True)
    os.makedirs(npz_ds_dir, exist_ok=True)
    os.makedirs(rgb_ds_dir, exist_ok=True)

    # Collect depth maps
    recursive = False
    targets = collect_tiffs(depth_dir, recursive)
    if not targets:
        print(f"No TIFF files found in {depth_dir}.", file=sys.stderr)
        sys.exit(2)

    # Parse orientations
    orient_map = parse_orientations(orient_path)

    done = 0
    skipped_no_orient = 0
    skipped_no_image = 0
    failed = 0

    json_rows: list[Dict[str, Any]] = []

    print(
        f"[i] Found {len(targets)} TIFF depth map(s).\n"
        f"  CSV  → {csv_dir}\n"
        f"  NPZ  → {npz_dir}\n"
        f"  RGB  → {rgb_dir}\n"
        f"  JSON → {os.path.join(out_root, 'data.json')}\n"
    )

    for idx, depth_path in enumerate(targets, 1):
        stem = os.path.splitext(os.path.basename(depth_path))[0]

        # Orientation must exist
        if stem not in orient_map:
            print(f"[{idx}/{len(targets)}] SKIP (no orientation) {stem}", file=sys.stderr)
            skipped_no_orient += 1
            continue

        # Image must exist
        img_src = find_image_for_stem(images_dir, stem)
        if img_src is None:
            print(f"[{idx}/{len(targets)}] SKIP (no matching image) {stem}", file=sys.stderr)
            skipped_no_image += 1
            continue

        try:
            arr = read_tiff_as_array(depth_path)

            # CSV + NPZ paths
            csv_path = os.path.join(csv_dir, f"{stem}.csv")
            npz_path = os.path.join(npz_dir, f"{stem}.npz")

            save_csv(csv_path, arr)
            save_npz(npz_path, arr)

            H, W = arr.shape
            ds_ratio = (downsample_pts / (H*W)) ** 0.5
            W_ds = int(ds_ratio*W)
            H_ds = int(ds_ratio*H)
            arr_ds = cv2.resize(arr, (W_ds, H_ds), interpolation=cv2.INTER_AREA)
            cimg = cv2.imread(img_src)
            cimg_ds = cv2.resize(cimg, (W_ds, H_ds), interpolation=cv2.INTER_AREA)

            csv_ds_path = os.path.join(csv_ds_dir, f"{stem}.csv")
            npz_ds_path = os.path.join(npz_ds_dir, f"{stem}.npz")
            rgb_ds_path = os.path.join(rgb_ds_dir, f"{stem}.png")
            save_csv(csv_ds_path, arr_ds)
            save_npz(npz_ds_path, arr_ds)
            cv2.imwrite(rgb_ds_path, cimg_ds)

            img_dst = os.path.join(rgb_dir, stem + ".png")
            cv2.imwrite(img_dst, cimg)
            
            # Copy image
            # img_ext = os.path.splitext(img_src)[1]
            # img_dst = os.path.join(rgb_dir, stem + img_ext)
            # shutil.copy2(img_src, img_dst)

            # Build JSON row
            orient = orient_map[stem]
            row = {
                "index": len(json_rows),
                "name": stem,
                "pose_utm": orient["pose_utm"],
                "agl": orient["pose_utm"][2],
                "OPK_deg": orient["OPK_deg"],
                "intrinsic": orient["intrinsic"],
            }
            json_rows.append(row)

            done += 1
            print(f"[{idx}/{len(targets)}] OK {stem}")

        except Exception as e:
            failed += 1
            print(f"[{idx}/{len(targets)}] FAILED {stem}: {e}", file=sys.stderr)

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
    print(f"  skipped (no orient): {skipped_no_orient}")
    print(f"  skipped (no image):  {skipped_no_image}")
    print(f"  failed:              {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
