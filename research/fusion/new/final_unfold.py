import os
import sys

import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")

from geom.surfaces import bspline_surface_mesh_from_ctrl
from fusion.fusion import reshape_in_polygon
from utils.conversion import pcd2pcm, pcdArr2pcd, pcm2pcd
from fusion.helper import NDFDrop_depth, NDFDrop_depth1, euclidean_distance_map, project_points_multi_fast, unfold_depth, unfold_depth1
from kinematics.clouds import apply_transform_points, orient_point_cloud_cgplane_global
from ioHandle.IOHandler import load_pcd, save_pcd
from kinematics.pose import Pose, RotFormat

import numpy as np
import open3d as o3d

def build_camera_rays(img_h, img_w, hfov_deg, pose, pyramidProj=False):
    """
    Build ray origins and directions in world/NWU frame.

    Origins are all at (0,0,0) and directions come from calcCameraDirs.

    Returns:
        origins: (N, 3)
        dirs:    (N, 3), unit-length directions
    """
    dirs = calcCameraDirs(
        shape=(img_h, img_w),
        hfov_deg=hfov_deg,
        pyramidProj=pyramidProj,
        pose=pose,
        do_rotate=True
    )  # (H, W, 3)
    dirs_flat = dirs.reshape(-1, 3)

    # Normalize directions
    norms = np.linalg.norm(dirs_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dirs_flat /= norms

    origins = np.zeros_like(dirs_flat)  # all rays start from origin (0,0,0)
    return origins, dirs_flat

def intersect_rays_with_spline(mesh: o3d.geometry.TriangleMesh,
                               origins,
                               dirs):
    """
    Intersect rays with the given triangle mesh using Open3D RaycastingScene.

    Returns:
        t_mesh: (N,) distances along ray to the first hit.
                For rays that miss, value is np.inf.
    """
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    origins = np.asarray(origins, dtype=float)
    dirs = np.asarray(dirs, dtype=float)
    rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
    rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays_o3d)
    t_hit = ans["t_hit"].numpy().reshape(-1)

    t_mesh = np.array(t_hit, dtype=float)
    t_mesh[(~np.isfinite(t_mesh)) | (t_mesh <= 0.0)] = np.inf
    return t_mesh

def fit_ctrl_grid_from_point_cloud(
    pcd: o3d.geometry.PointCloud,
    grid_w: int = 20,
    grid_h: int = 20,
    k_neighbors: int = 10,
    margin_scale: float = 1.2,
    idw: bool = True,
    idw_power: float = 2.0,
) -> np.ndarray:
    """
    Approximate a spline control grid by:
      - projecting to XY
      - sampling a regular XY grid across an *expanded* bounding box
      - for each grid point, estimating Z from nearest neighbors in the cloud
        (IDW by default; mean if idw=False)

    Args:
        pcd: Open3D point cloud
        grid_w, grid_h: control grid resolution in X (columns) and Y (rows)
        k_neighbors: K for the XY-nearest-neighbor lookup
        margin_scale: expand the XY bbox by this factor around its center
                      (e.g., 1.2 means +20% extents in both X and Y)
        idw: if True, use inverse-distance weighting for Z; else simple mean
        idw_power: power for IDW weights, typically 1..3

    Returns:
        (grid_h * grid_w, 3) flattened control points in XYZ.
    """
    # --- extract and sanitize points
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        raise ValueError("Point cloud is empty.")
    # drop non-finite
    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        raise ValueError("All points are NaN/Inf.")
    pts = pts[finite_mask]

    # --- XY bounding box with margin expansion about the center
    min_xy = pts[:, :2].min(axis=0)
    max_xy = pts[:, :2].max(axis=0)
    ctr_xy = 0.5 * (min_xy + max_xy)
    half_ext = 0.5 * (max_xy - min_xy)
    half_ext = half_ext * float(margin_scale)  # expand
    min_xy_ext = ctr_xy - half_ext
    max_xy_ext = ctr_xy + half_ext

    # --- sample a regular grid over the expanded XY box
    xs = np.linspace(min_xy_ext[0], max_xy_ext[0], grid_w)
    ys = np.linspace(min_xy_ext[1], max_xy_ext[1], grid_h)

    # --- KD-tree on XY only (set Z=0 so distances are purely XY)
    pts_xy = np.column_stack([pts[:, 0], pts[:, 1], np.zeros_like(pts[:, 2])])
    pcd_xy = o3d.geometry.PointCloud()
    pcd_xy.points = o3d.utility.Vector3dVector(pts_xy)
    kdtree = o3d.geometry.KDTreeFlann(pcd_xy)

    K = int(max(1, min(k_neighbors, pts.shape[0])))

    ctrl_grid = np.zeros((grid_h, grid_w, 3), dtype=float)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            query = np.array([x, y, 0.0], dtype=float)
            _, idx, d2 = kdtree.search_knn_vector_3d(query, K)
            # idx: neighbor indices into pts / pts_xy
            z_neighbors = pts[idx, 2]

            if idw:
                # inverse distance weighting in XY (use sqrt of squared distances)
                d = np.sqrt(np.asarray(d2, dtype=float))
                # if one neighbor is exactly at the query (d ~ 0), fall back to its Z
                near_zero = d < 1e-12
                if np.any(near_zero):
                    z_est = float(np.mean(z_neighbors[near_zero]))
                else:
                    w = 1.0 / np.power(d, idw_power)
                    z_est = float(np.dot(w, z_neighbors) / np.sum(w))
            else:
                # simple mean of neighbor Zs
                z_est = float(np.mean(z_neighbors))

            ctrl_grid[j, i, :] = [x, y, z_est]

    return ctrl_grid.reshape(-1, 3)

R = np.array([
    [-0.03030636,  0.99952340,  0.00586841],
    [ 0.92265970,  0.03023284, -0.38442826],
    [-0.38442248, -0.00623608, -0.92313623]
], dtype=float)
t = np.array([-0.7752736806869507, -20.579580307006836, 103.58663940429688], dtype=float)
T = np.eye(4, dtype=float)
T[:3, :3] = R
T[:3,  3] = t
pose = Pose(T, rot_format=RotFormat.W2C_ROT)  # camera at origin; directions rotated to NWU

# --------------- helpers ---------------
def calcCameraDirs(shape, hfov_deg, pyramidProj, pose, do_rotate):
    H, W = shape
    hfov_rad = np.radians(hfov_deg)
    f = (W / 2) / np.tan(hfov_rad / 2)
    cx, cy = W / 2, H / 2
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    X = (x_grid - cx) / f
    Y = (y_grid - cy) / f
    Z = np.ones_like(X)
    if not pyramidProj:
        norm = np.sqrt(X**2 + Y**2 + Z**2)  # unit-length in camera frame
        X /= norm; Y /= norm; Z /= norm
    dirs = np.stack((X, Y, Z), axis=-1)
    if do_rotate:
        # rotation preserves unit length => still unit after this
        return dirs @ pose.getCAM2NWU().T
    return dirs

def fit_plane_all_points(P):
    # least-squares plane ax+by+cz+d=0 over ALL points
    c = P.mean(axis=0)
    Q = P - c
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n /= np.linalg.norm(n)
    d = -np.dot(n, c)
    return (n[0], n[1], n[2], d), n, c

def compute_ratio_map(pcd_path, shape=(273,365), hfov_deg=62.0):
    # 1) load cloud and get zmin for the XY plane
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud failed to load or is empty: {pcd_path}")
    P = np.asarray(pcd.points)
    zmin = P[:, 2].min()

    # 2) best-fit plane to ALL points
    plane_abcd, n, centroid = fit_plane_all_points(P)
    a, b, c, d = plane_abcd
    n_vec = np.array([a, b, c], dtype=np.float64)

    # 3) per-pixel unit ray directions in NWU
    H, W = shape
    dirs = calcCameraDirs((H, W), hfov_deg, False, pose, True).astype(np.float64)  # (H,W,3), unit-length

    # 4) intersections (origin is 0)
    #    XY-like plane at z=zmin  => t_xy = zmin / d_z
    dz = dirs[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t_xy = zmin / dz

    #    fitted plane n·x + d = 0 with x = t * dir => t_fit = -d / (n·dir)
    denom = np.tensordot(dirs, n_vec, axes=([2], [0]))  # (H,W)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_fit = (-d) / denom

    # 5) validity masks: finite, denom!=0, and *in front* (t>0)
    valid_xy  = np.isfinite(t_xy)  & (dz    != 0) & (t_xy  > 0)
    valid_fit = np.isfinite(t_fit) & (denom != 0) & (t_fit > 0)
    valid_both = valid_xy & valid_fit

    # 6) distances from origin along unit rays => distance == |t|
    dist_xy  = np.where(valid_both, np.abs(t_xy), np.nan)
    dist_fit = np.where(valid_both, np.abs(t_fit), np.nan)

    # 7) ratio map
    ratio = dist_xy / dist_fit  # NaN / value stays NaN due to valid_both gating above

    # Optional: cast to float32 to save memory
    return ratio.astype(np.float32), {
        "zmin": float(zmin),
        "plane": {"a": float(a), "b": float(b), "c": float(c), "d": float(d)},
        "num_valid": int(np.count_nonzero(valid_both)),
        "total": int(H*W)
    }

def resacle_and_repose(pcd):
    pcd *= 113.86/30
    pcd[:,:, 0] += 65
    pcd[:,:, 1] += 21
    return pcd

def resacle_and_repose_rev(pcd):
    pcd[:,:, 1] -= 21
    pcd[:,:, 0] -= 65
    pcd /= 113.86/30
    return pcd

H, W = 273, 365
bg_radial = load_pcd("data/ortholoc/diffusion/rad_test/background/L08_R0000.pcd")
rgb = cv2.imread("data/ortholoc/projection/dp/rgb/L08_R0000.png")
bg_canonical = load_pcd("data/ortholoc/diffusion/can_test/background/L08_R0000.pcd")
prj_radial = load_pcd("data/ortholoc/projection/dp/radial/L08_R0000.pcd")
prj_canonical = load_pcd("data/ortholoc/projection/dp/canonical/L08_R0000.pcd")
gep = load_pcd("data/ortholoc/fusion/dp/ground/L08_R0000.pcd")
PCD_PATH = "data/ortholoc/diffusion/can_test/background/L08_R0000.pcd"
ratio_map, info = compute_ratio_map(PCD_PATH, shape=(H,W), hfov_deg=62.0)

prj_can_pcm = pcd2pcm(prj_canonical, H, W)
bg_rad = pcd2pcm(bg_radial, H, W)
pr_rad = pcd2pcm(prj_radial, H, W)
gep = pcd2pcm(gep, H, W)
dists = euclidean_distance_map(np.zeros_like(prj_can_pcm), prj_can_pcm)
new_bg_rad = bg_rad * (ratio_map[..., np.newaxis])
new_pr_rad = pr_rad * (ratio_map[..., np.newaxis])


new_proj = prj_can_pcm * (ratio_map[..., np.newaxis])
new_proj_pcd = pcm2pcd(new_proj)


new_scaled = resacle_and_repose(new_proj)
new_scaled = pcm2pcd(new_scaled)
save_pcd(np.asarray(new_proj_pcd.points), np.asarray(np.zeros_like(new_proj_pcd.points)), "new_proj_pcd.pcd")
save_pcd(np.asarray(new_scaled.points), np.asarray(np.zeros_like(new_scaled.points)), "new_scaled.pcd")


new_bg_rad_pcd = pcm2pcd(new_bg_rad)
pr_rad_pcd = pcm2pcd(new_pr_rad)
rev_gep = resacle_and_repose_rev(gep)
rev_gep[:,:,2] -= 10
gep_pcd = pcm2pcd(rev_gep)
save_pcd(np.asarray(new_bg_rad_pcd.points), np.asarray(np.zeros_like(new_bg_rad_pcd.points)), "new_bg_rad_pcd.pcd")
save_pcd(np.asarray(pr_rad_pcd.points), np.asarray(np.zeros_like(pr_rad_pcd.points)), "pr_rad_pcd.pcd")
save_pcd(np.asarray(gep_pcd.points), np.asarray(np.zeros_like(gep_pcd.points)), "gep_pcd.pcd")

# unfolded1 = unfold_depth1(pr_rad, bg_radial, resacle_and_repose_rev(gep), H, W)
# unfolded = unfold_depth(pr_rad, bg_radial, resacle_and_repose_rev(gep), H, W)

save_pcd(np.asarray(new_bg_rad_pcd.points), np.asarray(np.zeros_like(new_bg_rad_pcd.points)), "new_bg_rad_pcqd.pcd")
ndf, _ = NDFDrop_depth(pr_rad, bg_radial, rev_gep)

# ndf[:,:,2] -= 20.5
# unfolded[:,:,2] -= 15
# unfolded = pcm2pcd(resacle_and_repose(unfolded), visualization_image=rgb)
# unfolded1 = pcm2pcd(resacle_and_repose(unfolded1), visualization_image=rgb)

ndf = pcm2pcd(resacle_and_repose(ndf), visualization_image=rgb)
# save_pcd(np.asarray(unfolded.points), np.asarray(np.zeros_like(unfolded.points)), "unfolded.pcd")
# save_pcd(np.asarray(unfolded1.points), np.asarray(np.zeros_like(unfolded1.points)), "unfolded1.pcd")
save_pcd(np.asarray(ndf.points), np.asarray(np.zeros_like(ndf.points)), "ndf.pcd")


ctrl_flat = fit_ctrl_grid_from_point_cloud(
    bg_radial, grid_w=20, grid_h=20, k_neighbors=10
)
spline_mesh = bspline_surface_mesh_from_ctrl(
    ctrl_flat, grid_w=20, grid_h=20, su=200, sv=200
)

origins, dirs = build_camera_rays(
        img_h=H,
        img_w=W,
        hfov_deg=62,
        pose=pose,
        pyramidProj=False
    )
t_mesh = intersect_rays_with_spline(spline_mesh, origins, dirs)
mesh_hit_mask = np.isfinite(t_mesh) & (t_mesh > 0.0)
mesh_pts = origins[mesh_hit_mask] + dirs[mesh_hit_mask] * t_mesh[mesh_hit_mask, None]


mm = pcdArr2pcd(mesh_pts)
unfolded1 = unfold_depth(pr_rad, mm, ctrl_flat, resacle_and_repose_rev(gep), H, W)
ndf1, _ = NDFDrop_depth(pr_rad, ctrl_flat, rev_gep, H, W)

ndf1[:,:,2] -= 21
unfolded1[:,:,2] -= 21

unfolded1 = pcm2pcd(resacle_and_repose(unfolded1))
save_pcd(np.asarray(unfolded1.points), np.asarray(np.zeros_like(unfolded1.points)), "unfolded1.pcd")
ndf1 = pcm2pcd(resacle_and_repose(ndf1))
save_pcd(np.asarray(ndf1.points), np.asarray(np.zeros_like(ndf1.points)), "ndf1.pcd")

save_pcd(np.asarray(mm.points), np.asarray(np.zeros_like(mm.points)), "mm.pcd")

# cg_bg_canonical, T = orient_point_cloud_cgplane_global(bg_canonical)
# atp_bg_radial = apply_transform_points(np.asarray(bg_radial.points), T)
# atp_prj_radial = apply_transform_points(np.asarray(prj_radial.points), T)
# atp_prj_canonical = apply_transform_points(np.asarray(prj_canonical.points), T)

# save_pcd(np.asarray(cg_bg_canonical.points), np.asarray(np.zeros_like(cg_bg_canonical.points)), "cg_bg_canonical.pcd")
# save_pcd(atp_bg_radial, np.asarray(np.zeros_like(cg_bg_canonical.points)), "atp_bg_radial.pcd")
# save_pcd(atp_prj_radial, np.asarray(np.zeros_like(cg_bg_canonical.points)), "atp_prj_radial.pcd")


# atp_bg_radial_pcd = load_pcd("atp_bg_radial.pcd")
# atp_prj_radial_pcd = load_pcd("atp_prj_radial.pcd")
# atp_prj_radial_pcm = pcd2pcm(atp_prj_radial_pcd, H, W)
# unfolded = unfold_depth(atp_prj_radial_pcm, atp_bg_radial_pcd, atp_prj_radial_pcm, H, W)
# save_pcd(atp_prj_canonical, np.asarray(np.zeros_like(cg_bg_canonical.points)), "atp_prj_canonical.pcd")

# unfolded_pcd = pcm2pcd(unfolded)
# save_pcd(np.asarray(unfolded_pcd.points), np.asarray(np.zeros_like(unfolded_pcd.points)), "unfolded.pcd")

# gep_pcd_nwu = load_pcd("data/ortholoc/fusion/dp/ground/L08_R0000.pcd")
# gep_pcm_nwu = pcd2pcm(gep_pcd_nwu, H, W)

# iterlist = []
# a = gep_pcm_nwu[0,0,0]
# b = gep_pcm_nwu[0,0,1]
# iterlist.append((a,b))
# a = gep_pcm_nwu[0,W-1,0]
# b = gep_pcm_nwu[0,W-1,1]
# iterlist.append((a,b))
# a = gep_pcm_nwu[H-1,W-1,0]
# b = gep_pcm_nwu[H-1,W-1,1]
# iterlist.append((a,b))
# a = gep_pcm_nwu[H-1,0,0]
# b = gep_pcm_nwu[H-1,0,1]
# iterlist.append((a,b))
# unfolded_reshaped = reshape_in_polygon(iterlist, unfolded_pcd,H, W)

# atp_prj_canonical_pcd = pcm2pcd(atp_prj_canonical)
# atp_prj_canonical_reshaped = reshape_in_polygon(iterlist, atp_prj_canonical_pcd,H, W)

# save_pcd(np.asarray(unfolded_reshaped.points), np.asarray(np.zeros_like(unfolded_reshaped.points)), "unfolded_reshaped.pcd")
# save_pcd(np.asarray(atp_prj_canonical_reshaped.points), np.asarray(np.zeros_like(atp_prj_canonical_reshaped.points)), "atp_prj_canonical_reshaped.pcd")

