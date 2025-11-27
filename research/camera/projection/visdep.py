import numpy as np
import open3d as o3d
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from projection.helper import calcCameraDirs, project3D

# --- your functions are expected to be defined/imported in the same file/module ---
# def calcCameraDirs(shape, hfov_deg, pyramidProj, pose, do_rotate): ...
# def project3D(depth_img, pose, hfov_deg, scaling, bg=None, move=False, pyramidProj=False, do_rotate=True): ...

def _make_rays_lineset_from_dirs(dirs, stride=None, ray_len=1.0):
    """
    Create an Open3D LineSet of rays from the origin along 'dirs'.
    dirs: (H, W, 3) array of direction vectors
    stride: optional subsampling step. If None, choose automatically to keep ~3k rays.
    """
    H, W, _ = dirs.shape
    if stride is None:
        target = 3000
        stride = max(1, int(np.sqrt((H * W) / target)))
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    YY, XX = np.meshgrid(ys, xs, indexing='ij')
    sample_dirs = dirs[YY, XX].reshape(-1, 3)

    # Normalize directions and scale to ray_len
    norms = np.linalg.norm(sample_dirs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    endpoints = (sample_dirs / norms) * float(ray_len)

    # Build LineSet: a single shared origin (index 0) to each endpoint (1..N)
    points = np.vstack(([0.0, 0.0, 0.0], endpoints))
    lines = [[0, i + 1] for i in range(endpoints.shape[0])]
    colors = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(lines), 1))  # blue rays

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def _make_pcd_from_pc(pc, max_points=300000, color_by='z'):
    """
    pc: (H, W, 3) point cloud array
    max_points: subsample if larger than this
    color_by: 'z' or None
    """
    pts = pc.reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.size == 0:
        raise ValueError("Point cloud is empty after removing NaNs/Infs.")

    if max_points and pts.shape[0] > max_points:
        step = int(np.ceil(pts.shape[0] / max_points))
        pts = pts[::step]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    if color_by == 'z':
        z = pts[:, 2]
        # Normalize to [0,1] for simple gradient (yellow→blue)
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        denom = (zmax - zmin) if np.isfinite(zmax - zmin) and (zmax - zmin) > 1e-12 else 1.0
        zn = (z - zmin) / denom
        colors = np.stack([zn, np.ones_like(zn) - zn * 0.5, np.ones_like(zn) - zn], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_dirs_and_pc(
    depth_img,
    pose,
    hfov_deg,
    bg=None,
    move=False,
    pyramidProj=False,
    do_rotate=False,
    ray_len=1.0,
    ray_stride=None,
    max_points=300000,
):
    """
    Computes dirs & pc using YOUR functions, then visualizes both with Open3D.
    """
    # 1) Compute direction field in the requested frame
    dirs = calcCameraDirs(depth_img.shape, hfov_deg, pyramidProj, pose, do_rotate)

    # 2) Compute point cloud using your project3D
    pc, _ = project3D(depth_img, pose, hfov_deg, bg=bg, move=move,
                      pyramidProj=pyramidProj, do_rotate=do_rotate)

    # 3) Build Open3D geometries
    rays = _make_rays_lineset_from_dirs(dirs, stride=ray_stride, ray_len=150)
    pcd = _make_pcd_from_pc(pc, max_points=max_points, color_by='z')

    # 4) Axis frame for reference (NWU or camera depending on do_rotate)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=ray_len * 1.5, origin=[0, 0, 0])

    # 5) Show
    o3d.visualization.draw_geometries(
        [axis, rays, pcd],
        window_name="Dirs (rays from origin) + Point Cloud",
        width=1280, height=800
    )

# ----------------------------
# Example usage (optional demo)
# ----------------------------
if __name__ == "__main__":
    # This demo assumes you already have:
    # - calcCameraDirs and project3D defined (with calc_scale_factor available)
    # - a valid 'pose' object implementing pose.getCAM2NWU() and pose.p6.{x,y,z}
    #
    # Replace the following placeholders with your real data.
    
    depth_img = np.loadtxt("data/usegeo_light/data/metric_depth/depth_pro/2021-04-23_13-17-12_S2223314_DxO.csv", delimiter=',', dtype=np.float64)
    depth_img = depth_img[:500, :]
    H, W = depth_img.shape

    # ---- Example pose stub (identity rotation, origin). Replace with your real pose. ----
    class _P6:  # simple container for position
        def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z

    class _PoseStub:
        def __init__(self): self.p6 = _P6(0.0, 0.0, 0.0)
        def getCAM2NWU(self): return np.eye(3, dtype=float)

    pose = _PoseStub()

    # Visualize (adjust ray_len/stride if it feels heavy/light)
    visualize_dirs_and_pc(
        depth_img=depth_img,
        pose=pose,
        hfov_deg=62.0,
        bg=None,
        move=False,
        pyramidProj=False,
        do_rotate=True,
        ray_len=1.0,
        ray_stride=None,      # auto stride to ~3k rays
        max_points=200000,
    )
