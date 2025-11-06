#!/usr/bin/env python3
# fuse_xy_keep_z.py
import argparse
import copy
import numpy as np
import open3d as o3d


def load_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Failed to read points from: {path}")
    return pcd


def assert_one_to_one(p1: o3d.geometry.PointCloud, p2: o3d.geometry.PointCloud):
    n1 = np.asarray(p1.points).shape[0]
    n2 = np.asarray(p2.points).shape[0]
    assert n1 == n2, f"Point-count mismatch: pcd1={n1}, pcd2={n2}. No 1–1 mapping."


def fuse_xy_keep_z(pcd1: o3d.geometry.PointCloud,
                   pcd2: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Create pcd3 where:
      pcd3.x, pcd3.y = pcd2.x, pcd2.y
      pcd3.z          = pcd1.z
    """
    assert_one_to_one(pcd1, pcd2)
    pts1 = np.asarray(pcd1.points)  # (N,3)
    pts2 = np.asarray(pcd2.points)  # (N,3)

    fused = np.empty_like(pts1)
    fused[:, :2] = pts2[:, :2]
    fused[:, 2] = pts1[:, 2]

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(fused)

    # If you prefer to carry over colors, uncomment ONE of these:
    # pcd3.colors = copy.deepcopy(pcd1.colors)  # keep colors from pcd1
    # pcd3.colors = copy.deepcopy(pcd2.colors)  # or keep colors from pcd2
    return pcd3


def tint_copy(pcd: o3d.geometry.PointCloud, rgb):
    q = copy.deepcopy(pcd)
    if len(q.colors) == 0:
        q.paint_uniform_color(rgb)
    else:
        # Lightly blend existing colors toward the tint
        c = np.asarray(q.colors)
        c = 0.6 * c + 0.4 * np.array(rgb)[None, :]
        q.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1))
    return q


def side_by_side(pcds, gap_scale=1.25):
    """
    Returns translated copies placed along +X with automatic spacing.
    """
    placed = []
    offset_x = 0.0
    base_extent = None
    for i, p in enumerate(pcds):
        q = copy.deepcopy(p)
        if i == 0:
            # Use first cloud extent to set spacing scale
            bbox = q.get_axis_aligned_bounding_box()
            ext = bbox.get_extent()
            base_extent = float(max(ext[0], ext[1], ext[2]))
            if base_extent <= 0:
                base_extent = 1.0
        q.translate([offset_x, 0.0, 0.0])
        placed.append(q)
        offset_x += gap_scale * base_extent
    return placed


def visualize_three(p1, p2, p3, mode="overlay"):
    # Color them distinctly
    a = tint_copy(p1, [0.98, 0.64, 0.18])  # amber
    b = tint_copy(p2, [0.20, 0.70, 1.00])  # blue
    c = tint_copy(p3, [0.10, 0.95, 0.35])  # green

    geoms = [a, b, c]
    title = "Overlay: pcd1=amber, pcd2=blue, pcd3=green"
    if mode == "sbs":
        geoms = side_by_side(geoms, gap_scale=1.35)
        title = "Side-by-side (left→right): pcd1=amber, pcd2=blue, pcd3=green"

    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1280,
        height=800,
        point_show_normal=False,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Fuse XY from pcd2 with Z from pcd1 → pcd3; visualize the three."
    )
    ap.add_argument("pcd1", help="Path to point cloud 1 (provides Z)")
    ap.add_argument("pcd2", help="Path to point cloud 2 (provides X,Y)")
    ap.add_argument("--out", "-o", help="Optional path to save pcd3 (format inferred)")
    ap.add_argument(
        "--view",
        choices=["overlay", "sbs"],
        default="overlay",
        help="Visualization: overlay (default) or sbs (side-by-side).",
    )
    args = ap.parse_args()

    p1 = load_cloud(args.pcd1)
    p2 = load_cloud(args.pcd2)
    p3 = fuse_xy_keep_z(p1, p2)

    if args.out:
        ok = o3d.io.write_point_cloud(args.out, p3)
        if not ok:
            raise RuntimeError(f"Failed to write fused cloud to: {args.out}")
        print(f"[saved] pcd3 → {args.out}")

    visualize_three(p1, p2, p3, mode=args.view)


if __name__ == "__main__":
    main()
