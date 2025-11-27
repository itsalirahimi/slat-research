#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def clean_and_downsample(pcd, voxel_size=None):
    """
    Remove NaN/inf and optionally voxel-downsample.
    Handles Open3D API differences for remove_non_finite_points().
    """
    # --- remove non-finite ---
    clean_result = pcd.remove_non_finite_points()
    if isinstance(clean_result, tuple):
        pcd, _ = clean_result
    else:
        pcd = clean_result

    if len(pcd.points) == 0:
        raise ValueError("Point cloud has no valid points after cleaning.")

    # --- auto voxel size if not provided ---
    if voxel_size is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())
        # heuristic: ~300 voxels across the diagonal
        voxel_size = diag / 300.0 if diag > 0 else 0.01

    if voxel_size > 0:
        print(f"[info] Voxel downsampling with size = {voxel_size:.5f}")
        pcd = pcd.voxel_down_sample(voxel_size)

    print("[info] Points after cleaning/downsample:", np.asarray(pcd.points).shape[0])
    return pcd, voxel_size


def estimate_normals_fast(pcd, voxel_size, max_nn=30):
    """
    Estimate normals quickly using a radius tied to voxel size,
    and orient them roughly upward (Z+) for speed.
    """
    if len(pcd.points) == 0:
        raise ValueError("Point cloud has no points for normal estimation.")

    # radius a bit larger than voxel size
    radius = voxel_size * 2.5
    print(f"[info] Estimating normals with radius = {radius:.5f}")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # Fast orientation: align with +Z instead of consistent tangent plane
    pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    return radius


def mesh_from_bpa_fast(pcd, pivot_radius):
    """
    Create mesh using Ball Pivoting Algorithm.
    Uses downsampled points as vertices → colors map directly.
    """
    radii = o3d.utility.DoubleVector([pivot_radius, pivot_radius * 2.0])
    print(f"[info] BPA radii: {list(radii)}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii
    )

    if pcd.has_colors():
        # BPA uses the same vertices as the input PCD
        mesh.vertex_colors = pcd.colors
    else:
        print("[warn] Input point cloud has no colors; mesh will be gray.")

    mesh.compute_vertex_normals()
    return mesh


def mesh_from_poisson_slow_but_nice(pcd, depth=9, density_q=0.01):
    """
    Optional: Poisson reconstruction (slower, smoother).
    Still here as an option, but not the 'fast' way.
    """
    print(f"[info] Running Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    densities = np.asarray(densities)
    print("[info] Mesh vertices before pruning:", np.asarray(mesh.vertices).shape[0])

    thr = np.quantile(densities, density_q)
    vertices_to_remove = densities < thr
    mesh = mesh.remove_vertices_by_mask(vertices_to_remove)

    print("[info] Mesh vertices after pruning:", np.asarray(mesh.vertices).shape[0])

    # Transfer color from original PCD via nearest neighbor
    if pcd.has_colors():
        print("[info] Transferring colors from PCD to mesh vertices (this is slower)...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_colors = np.asarray(pcd.colors)
        mesh_vertices = np.asarray(mesh.vertices)

        # simple Python loop; acceptable for moderate sizes
        new_colors = np.zeros_like(mesh_vertices)
        for i, v in enumerate(mesh_vertices):
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            new_colors[i] = pcd_colors[idx[0]]

        mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    else:
        print("[warn] Input point cloud has no colors; mesh will be gray.")

    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="FAST: PCD → Mesh (Open3D, keep colors) and visualize."
    )
    parser.add_argument("pcd_path", type=str, help="Input .pcd file")
    parser.add_argument(
        "--method",
        choices=["bpa", "poisson"],
        default="bpa",
        help="Meshing method (bpa=fast, poisson=smoother but slower)",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=None,
        help="Voxel size for downsampling (default: auto from scene size)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output mesh file (.ply / .obj / .stl / etc.)",
    )
    args = parser.parse_args()

    pcd_path = Path(args.pcd_path)
    if not pcd_path.is_file():
        raise FileNotFoundError(f"Cannot find input PCD: {pcd_path}")

    print(f"[info] Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    print("[info] Raw points:", np.asarray(pcd.points).shape[0])

    # Clean + downsample (big speed win)
    pcd, voxel_size = clean_and_downsample(pcd, voxel_size=args.voxel)

    # Normals
    pivot_radius = estimate_normals_fast(pcd, voxel_size)

    # Meshing
    if args.method == "bpa":
        print("[info] Using FAST Ball Pivoting meshing...")
        mesh = mesh_from_bpa_fast(pcd, pivot_radius)
    else:
        print("[info] Using Poisson meshing (slower)...")
        mesh = mesh_from_poisson_slow_but_nice(pcd, depth=9, density_q=0.01)

    # Save if requested
    if args.save is not None:
        out_path = Path(args.save)
        print(f"[info] Saving mesh to: {out_path}")
        o3d.io.write_triangle_mesh(str(out_path), mesh)

    print("[info] Visualizing mesh...")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="PCD → Mesh (FAST)",
        width=1280,
        height=720,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
