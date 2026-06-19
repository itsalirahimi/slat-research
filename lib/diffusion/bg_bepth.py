
import numpy as np
import open3d as o3d
import imageio.v2 as imageio
import copy


# ------------------ meshing ------------------

def pcd_to_mesh_poisson(
    pcd: o3d.geometry.PointCloud,
    depth: int = 9,
    normal_radius: float = None,
    normal_max_nn: int = 30,
    density_quantile_keep: float = 0.01,
):
    if normal_radius is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_extent())
        normal_radius = max(diag * 0.01, 1e-3)

    pcd = pcd.voxel_down_sample(voxel_size=normal_radius * 0.5)
    bbox = pcd.get_axis_aligned_bounding_box()

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)

    if densities.size == len(mesh.vertices) and densities.size > 0:
        thr = np.quantile(densities, density_quantile_keep)
        mesh.remove_vertices_by_mask((densities < thr).tolist())

    mesh = mesh.crop(bbox)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def dilate_mesh_along_normals(mesh: o3d.geometry.TriangleMesh, offset: float):
    """Extrapolate mesh outward along vertex normals (good for closing tiny holes / boundary misses)."""
    m = copy.deepcopy(mesh)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices, dtype=np.float32)
    n = np.asarray(m.vertex_normals, dtype=np.float32)
    m.vertices = o3d.utility.Vector3dVector((v + offset * n).astype(np.float64))
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_non_manifold_edges()
    m.compute_vertex_normals()
    return m


def build_convex_hull_fallback(pcd: o3d.geometry.PointCloud,
                            hull_margin: float = 1.01,
                            simplify_to_tris: int = 80000):
    hull, _ = pcd.compute_convex_hull()
    hull.remove_degenerate_triangles()
    hull.remove_duplicated_triangles()
    hull.remove_duplicated_vertices()
    hull.remove_non_manifold_edges()
    hull.scale(hull_margin, center=hull.get_center())

    if simplify_to_tris is not None and len(hull.triangles) > simplify_to_tris:
        hull = hull.simplify_quadric_decimation(target_number_of_triangles=simplify_to_tris)

    hull.compute_vertex_normals()
    return hull


def build_enclosure_sphere(ref_mesh: o3d.geometry.TriangleMesh,
                        cam_origin_world: np.ndarray,
                        margin: float = 1.15,
                        min_radius: float = 1.0,
                        resolution: int = 30):
    cam_origin_world = np.asarray(cam_origin_world, dtype=np.float32).reshape(3,)
    bbox = ref_mesh.get_axis_aligned_bounding_box()
    corners = np.asarray(bbox.get_box_points(), dtype=np.float32)
    r = float(np.linalg.norm(corners - cam_origin_world[None, :], axis=1).max())
    r = max(r * margin, float(min_radius))

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=resolution)
    sphere.translate(cam_origin_world)
    sphere.compute_vertex_normals()
    return sphere


# ------------------ ray prep + casting ------------------

def _prep_rays(dirs_hw3: np.ndarray, cam_origin, pose_R=None, pose_t=None):
    H, W, _ = dirs_hw3.shape
    N = H * W

    dirs = dirs_hw3.reshape(N, 3).astype(np.float32)
    origin = np.array(cam_origin, dtype=np.float32).reshape(1, 3)

    if pose_R is not None:
        R = np.asarray(pose_R, dtype=np.float32).reshape(3, 3)
        dirs = (R @ dirs.T).T
    if pose_t is not None:
        t = np.asarray(pose_t, dtype=np.float32).reshape(1, 3)
        origin = origin + t

    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_unit = dirs / np.clip(norms, 1e-8, None)

    origins = np.repeat(origin, N, axis=0)
    rays = np.concatenate([origins, dirs_unit], axis=1).astype(np.float32)  # (N,6)
    return H, W, rays, origins, dirs_unit, origin.squeeze(0)


def _cast_single(mesh_legacy: o3d.geometry.TriangleMesh, rays_np: np.ndarray):
    scene = o3d.t.geometry.RaycastingScene()
    gid = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy))
    ans = scene.cast_rays(o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32))
    t_hit = ans["t_hit"].numpy().astype(np.float32)
    geom_ids = ans["geometry_ids"].numpy().astype(np.int32)
    hit = np.isfinite(t_hit) & (geom_ids == gid)
    return t_hit, hit


def fill_missing_nearest(depth_hw1: np.ndarray, valid_hw: np.ndarray):
    """
    Nearest-neighbor fill using distance transform (fills invalid pixels with closest valid depth).
    valid_hw: (H,W) boolean, True where depth is trusted.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return depth_hw1  # skip if scipy not available

    d = depth_hw1[..., 0].astype(np.float32)
    invalid = ~valid_hw
    if not invalid.any():
        return depth_hw1

    # distance to nearest valid (zero) => pass invalid mask (1=invalid, 0=valid)
    _, (iy, ix) = distance_transform_edt(invalid, return_indices=True)
    filled = d[iy, ix]
    return filled[..., None].astype(np.float32)


def cast_dirs_multistage(
    dirs_hw3: np.ndarray,
    bg_mesh: o3d.geometry.TriangleMesh,
    bg_mesh_dilated: o3d.geometry.TriangleMesh,
    hull_mesh: o3d.geometry.TriangleMesh,
    cam_origin=(0.0, 0.0, 0.0),
    pose_R=None,
    pose_t=None,
    use_sphere=True,
):
    """
    Stages:
    0 -> bg_mesh
    1 -> dilated bg_mesh (extrapolated)
    2 -> convex hull
    3 -> enclosure sphere
    Returns:
    hit_points_hw3, distances_hw1, src_hw1 (uint8 labels)
    """
    H, W, rays, origins, dirs_unit, origin_world = _prep_rays(
        dirs_hw3, cam_origin, pose_R=pose_R, pose_t=pose_t
    )
    N = H * W

    t_out = np.full((N,), np.inf, dtype=np.float32)
    src = np.full((N,), 255, dtype=np.uint8)

    # Stage 0: bg mesh
    t0, hit0 = _cast_single(bg_mesh, rays)
    t_out[hit0] = t0[hit0]
    src[hit0] = 0

    # Stage 1: dilated bg mesh for remaining misses
    miss = ~np.isfinite(t_out)
    if miss.any():
        t1, hit1 = _cast_single(bg_mesh_dilated, rays[miss])
        miss_idx = np.flatnonzero(miss)
        fill = miss_idx[hit1]
        t_out[fill] = t1[hit1]
        src[fill] = 1

    # Stage 2: convex hull
    miss = ~np.isfinite(t_out)
    if miss.any():
        t2, hit2 = _cast_single(hull_mesh, rays[miss])
        miss_idx = np.flatnonzero(miss)
        fill = miss_idx[hit2]
        t_out[fill] = t2[hit2]
        src[fill] = 2

    # Stage 3: sphere (guarantee)
    miss = ~np.isfinite(t_out)
    if use_sphere and miss.any():
        ref = hull_mesh if hull_mesh is not None else bg_mesh
        sphere = build_enclosure_sphere(ref, origin_world)
        t3, hit3 = _cast_single(sphere, rays[miss])
        miss_idx = np.flatnonzero(miss)
        fill = miss_idx[hit3]
        t_out[fill] = t3[hit3]
        src[fill] = 3

    hit_pts = origins + dirs_unit * t_out[:, None]
    hit_pts[~np.isfinite(t_out)] = np.nan

    return hit_pts.reshape(H, W, 3), t_out.reshape(H, W, 1), src.reshape(H, W, 1)


# ------------------ saving + viz ------------------

def save_depth_outputs_no_white(
    distances_hw1: np.ndarray,
    npy_path: str,
    depth_png_path: str,
    clip_percentiles=(0.5, 99.95),
    log_compress: bool = True,
):
    """
    - Saves .npy as-is
    - Saves visualization PNG without pure-white saturation:
        * robust clip
        * optional log compression
        * map to 0..254 (never 255)
    """
    np.save(npy_path, distances_hw1)

    d = distances_hw1[..., 0].astype(np.float32)
    finite = np.isfinite(d)

    lo, hi = np.percentile(d[finite], clip_percentiles)
    hi = max(float(hi), float(lo) + 1e-6)

    d_clip = np.clip(d, lo, hi)
    if log_compress:
        norm = np.log1p(d_clip - lo) / np.log1p(hi - lo)
    else:
        norm = (d_clip - lo) / (hi - lo)

    norm[~finite] = 0.0
    img_u8 = (np.clip(norm, 0.0, 0.999) * 254.0).astype(np.uint8)  # <- never pure white
    imageio.imwrite(depth_png_path, img_u8)
    return img_u8


def visualize_intersections(bg_mesh, hit_points_hw3, src_hw1, stride=6):
    pts = hit_points_hw3[::stride, ::stride, :].reshape(-1, 3)
    src = src_hw1[::stride, ::stride, 0].reshape(-1)

    valid = np.all(np.isfinite(pts), axis=1)
    pts, src = pts[valid], src[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    colors = np.zeros((len(pts), 3), dtype=np.float64)
    colors[src == 0] = np.array([0.1, 0.9, 0.1])  # bg
    colors[src == 1] = np.array([0.2, 0.7, 1.0])  # dilated bg
    colors[src == 2] = np.array([0.3, 0.3, 1.0])  # hull
    colors[src == 3] = np.array([0.9, 0.1, 0.1])  # sphere
    pcd.colors = o3d.utility.Vector3dVector(colors)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([bg_mesh, pcd, frame])


# ------------------ YOUR PIPELINE ------------------

dirs = calcCameraDirs(metric_depth_ds.shape, self.config.hfov_deg, False, pose, True)
bg_pcd = load_pcd("data/bvc/diffusion/test/canonical/background/seq39_000000.pcd")

assert dirs.shape[:2] == metric_depth_ds.shape[:2]
dirs = dirs.astype(np.float32)

# 1) pcd -> mesh
bg_mesh = pcd_to_mesh_poisson(bg_pcd, depth=9)

# 1b) extrapolate mesh (fix tiny holes / boundary misses)
bbox = bg_mesh.get_axis_aligned_bounding_box()
diag = float(np.linalg.norm(bbox.get_extent()))
dilate_offset = max(diag * 0.002, 1e-3)  # tweak: 0.001..0.005 typically
bg_mesh_dil = dilate_mesh_along_normals(bg_mesh, dilate_offset)

# 1c) hull fallback (bigger misses)
bg_hull = build_convex_hull_fallback(bg_pcd, hull_margin=1.01, simplify_to_tris=80000)

# 2-3) multistage intersect
intersected_points, distances, src = cast_dirs_multistage(
    dirs_hw3=dirs,
    bg_mesh=bg_mesh,
    bg_mesh_dilated=bg_mesh_dil,
    hull_mesh=bg_hull,
    cam_origin=(0.0, 0.0, 0.0),
    pose_R=None,
    pose_t=None,
    use_sphere=True,
)

# 3b) IMPORTANT: remove remaining “white specks” by replacing sphere pixels with nearest valid neighbor
# (sphere pixels are the usual cause of far-depth spikes)
valid = (src[..., 0] != 3)  # treat sphere as invalid for filling
distances_filled = fill_missing_nearest(distances, valid_hw=valid)

# Save debug masks
imageio.imwrite("bg_src_mask.png", src[..., 0])  # 0/1/2/3 labels

# 4) save distances + points (use filled distances as final)
np.save("bg_depth_distances.npy", distances_filled)
np.save("bg_intersection_points.npy", intersected_points)

# 5) export normalized depth image (no pure white)
save_depth_outputs_no_white(
    distances_hw1=distances_filled,
    npy_path="bg_depth_distances.npy",
    depth_png_path="bg_depth_normalized.png",
    clip_percentiles=(0.5, 99.95),
    log_compress=True,
)

print("Saved: bg_depth_distances.npy bg_depth_normalized.png bg_intersection_points.npy bg_src_mask.png")

# Visualization (green=bg, cyan=dilated, blue=hull, red=sphere)
visualize_intersections(bg_mesh, intersected_points, src, stride=6)
exit()



