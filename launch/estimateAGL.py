#!/usr/bin/env python3
"""
Accurate camera AGL from a PCD:
  (1) Best surface-normal intercept to the camera point (refined via local plane)
  (2) Vertical (Z-axis) drop to the local surface plane
  + Visualization

Usage:
  python cam_agl_precise.py gt.pcd --point X Y Z
Optional:
  --knorm 40        # K for normal estimation
  --cand 8192       # #nearest candidates around camera to test
  --kplane 400      # K for local plane fit near the best surface point
  --radius 0        # Hybrid radius for normals (0=disabled)
"""

import argparse
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

# ---------------------- Utilities ----------------------

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"[ERROR] Empty point cloud: {path}")
    return pcd

def bbox_diag(pcd: o3d.geometry.PointCloud) -> float:
    aabb = pcd.get_axis_aligned_bounding_box()
    return np.linalg.norm(np.asarray(aabb.get_max_bound()) - np.asarray(aabb.get_min_bound()))

def estimate_normals(pcd: o3d.geometry.PointCloud, knorm: int, radius: float = 0.0) -> None:
    if radius and radius > 0:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knorm))
    else:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knorm))
    nrm = np.asarray(pcd.normals, dtype=np.float64)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    pcd.normals = o3d.utility.Vector3dVector(nrm)

def fit_plane(points: np.ndarray, trim_ratio: float = 0.2, iters: int = 2):
    """
    Robust local plane via iterated trimmed PCA.
    Returns (center, unit_normal).
    """
    P = points
    c = P.mean(axis=0)
    for _ in range(max(1, iters)):
        A = P - c
        w, v = np.linalg.eigh(A.T @ A)
        n = v[:, 0]
        n = n / (np.linalg.norm(n) + 1e-12)
        # distances to plane
        d = np.abs((P - c) @ n)
        keep = int(max(8, (1.0 - trim_ratio) * len(P)))
        idx = np.argpartition(d, keep-1)[:keep]
        P = P[idx]
        c = P.mean(axis=0)
    # final normal
    A = P - c
    w, v = np.linalg.eigh(A.T @ A)
    n = v[:, 0]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n

def line_signed_t_and_residual(cam: np.ndarray, p0: np.ndarray, n: np.ndarray):
    """
    For the normal line L(t)=p0 + t n (||n||=1), compute:
      t  = signed distance along n from p0 to orthogonal projection of cam onto L
      r  = residual distance from cam to L
    """
    v = cam - p0
    t = float(np.dot(v, n))
    r = float(np.linalg.norm(v - t * n))
    return t, r

def vertical_intersection_with_plane(cam, c_plane, n_plane):
    """
    Intersect vertical line (x=cam.x, y=cam.y) with plane: n·(X - c) = 0.
    Returns intersection point or None if nz ~ 0.
    """
    nx, ny, nz = n_plane
    if abs(nz) < 1e-12:
        return None
    x0, y0, z0 = c_plane
    z = z0 - (nx*(cam[0]-x0) + ny*(cam[1]-y0)) / nz
    return np.array([cam[0], cam[1], z], dtype=np.float64)

def incidence_angle_deg(cam_vec: np.ndarray, n: np.ndarray):
    """
    Angle between the direction from surface to camera and the surface normal.
    0° means camera lies exactly on the normal direction.
    """
    a = cam_vec / (np.linalg.norm(cam_vec) + 1e-12)
    b = n / (np.linalg.norm(n) + 1e-12)
    cosv = np.clip(np.abs(np.dot(a, b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

# ---------------------- Visualization helpers ----------------------

def make_line(p, q, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([p, q]))
    ls.lines  = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls

def make_sphere(center, radius, color):
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sp.compute_vertex_normals()
    sp.translate(center)
    sp.paint_uniform_color(color)
    return sp

def make_plane_patch(center, normal, size):
    n = normal / (np.linalg.norm(normal) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, tmp); t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1);  t2 /= np.linalg.norm(t2) + 1e-12
    s = size * 0.5
    corners = np.array([
        center + s*( t1 + t2),
        center + s*( t1 - t2),
        center + s*(-t1 - t2),
        center + s*(-t1 + t2),
    ])
    tri = o3d.geometry.TriangleMesh()
    tri.vertices = o3d.utility.Vector3dVector(corners)
    tri.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[0,2,3]], dtype=np.int32))
    tri.compute_vertex_normals()
    tri.paint_uniform_color([0.2, 1.0, 0.6])
    return tri

# ---------------------- Core procedure ----------------------

def compute_agl_and_visualize(pcd_path: str, cam_xyz, knorm=40, cand=8192, kplane=400, radius=0.0):
    # Load and prep
    pcd = load_pcd(pcd_path)
    D   = bbox_diag(pcd)
    pts = np.asarray(pcd.points, dtype=np.float64)
    cam = np.array(cam_xyz, dtype=np.float64)

    # Normals (global)
    estimate_normals(pcd, knorm=knorm, radius=radius)

    # Orient normals toward camera for sign consistency
    try:
        pcd.orient_normals_towards_camera_location(cam.tolist())
    except Exception:
        pass

    normals = np.asarray(pcd.normals, dtype=np.float64)

    # KD-trees
    tree3 = cKDTree(pts)
    tree2 = cKDTree(pts[:, :2])

    # Candidate set near camera
    K = min(int(cand), len(pts))
    _, cand_idx = tree3.query(cam, k=K)
    if K == 1:
        cand_idx = np.array([cand_idx], dtype=int)

    # For each candidate point, compute residual to its normal line
    best = (None, np.inf, 0.0)  # (index, residual, t_signed)
    for i in cand_idx:
        p0 = pts[i]
        n0 = normals[i] / (np.linalg.norm(normals[i]) + 1e-12)
        t, r = line_signed_t_and_residual(cam, p0, n0)
        if r < best[1]:
            best = (int(i), r, t)

    idx_star, residual_line, t_raw = best
    p_star = pts[idx_star]

    # Refine: robust plane fit in neighborhood around p_star, recompute line using plane normal
    kplane = min(int(kplane), len(pts))
    _, nbr_idx = tree3.query(p_star, k=kplane)
    nbrs = pts[nbr_idx] if np.ndim(nbr_idx) else pts[np.newaxis, :]
    c_plane, n_plane = fit_plane(nbrs, trim_ratio=0.25, iters=3)

    # Prefer "up" normal
    if n_plane[2] < 0:
        n_plane = -n_plane

    # Recompute signed t and residual using plane normal line through p_star
    t_refined, residual_refined = line_signed_t_and_residual(cam, p_star, n_plane)

    # AGL along surface normal (refined)
    agl_normal = abs(t_refined)

    # Camera incidence angle (0° = exactly on normal)
    ang_deg = incidence_angle_deg(cam - p_star, n_plane)

    # Vertical Z-drop using the local plane
    gz = vertical_intersection_with_plane(cam, c_plane, n_plane)
    if gz is None:
        # fallback: nearest by XY
        _, j = tree2.query(cam[:2], k=1)
        gz = np.array([cam[0], cam[1], pts[j, 2]], dtype=np.float64)
    agl_vertical = float(abs(cam[2] - gz[2]))

    # --------- Print
    print("\n=== Accurate AGL Report ===")
    print(f"Best surface point index: {idx_star}")
    print(f"Best surface point p*: {p_star}")
    print(f"Refined plane center: {c_plane}")
    print(f"Refined plane normal: {n_plane}")
    print(f"[Normal AGL]  distance_along_normal = {agl_normal:.6f}  (signed t = {t_refined:.6f})")
    print(f"[Normal AGL]  residual_to_normal_line = {residual_refined:.6f} (after plane refinement)")
    print(f"[Incidence]   angle(camera_dir, normal) = {ang_deg:.3f} deg (0° ⇒ on normal)")
    print(f"[Vertical AGL]ΔZ to plane under camera (Z-axis drop) = {agl_vertical:.6f}\n")

    # --------- Viz
    geoms = [pcd]
    r_cam = max(D * 0.012, 1e-3)
    r_pt  = r_cam * 0.8

    geoms.append(make_sphere(cam, r_cam, (0.1, 0.6, 1.0)))       # camera
    geoms.append(make_sphere(p_star, r_pt, (1.0, 0.3, 0.3)))     # best point

    # Normal line (through p_star, plane normal direction); show the segment from p_star to cam’s foot on line
    cam_on_line = p_star + t_refined * n_plane
    geoms.append(make_line(p_star, cam_on_line, (1.0, 0.0, 0.0)))  # red

    # Local plane patch
    geoms.append(make_plane_patch(c_plane, n_plane, size=D * 0.08))

    # Vertical (Z-axis) drop
    geoms.append(make_line(cam, gz, (0.0, 0.8, 0.0)))              # green
    geoms.append(make_sphere(gz, r_pt*0.9, (0.2, 1.0, 0.2)))

    # Show a short normal arrow at p_star
    geoms.append(make_line(p_star, p_star + n_plane * (D * 0.06), (0.4, 0.4, 0.4)))

    # Camera frame
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=D*0.03, origin=cam))

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Accurate AGL: Surface-Normal & Vertical (Z) Lines",
        width=1400, height=900
    )

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="Accurate camera AGL from PCD via surface normal and Z-axis")
    ap.add_argument("--pcd", help="Path to gt.pcd")
    ap.add_argument("--point", "-p", nargs=3, type=float, required=True, metavar=("X","Y","Z"),
                    help="Camera point coordinates")
    ap.add_argument("--knorm", type=int, default=40, help="K for normal estimation")
    ap.add_argument("--cand",  type=int, default=8192, help="Candidate points near camera to test normals")
    ap.add_argument("--kplane", type=int, default=400, help="K for local plane fit around best point")
    ap.add_argument("--radius", type=float, default=0.0, help="Hybrid radius for normal estimation (0 disables)")
    args = ap.parse_args()

    compute_agl_and_visualize(
        pcd_path=args.pcd,
        cam_xyz=args.point,
        knorm=args.knorm,
        cand=args.cand,
        kplane=args.kplane,
        radius=args.radius,
    )

if __name__ == "__main__":
    main()
