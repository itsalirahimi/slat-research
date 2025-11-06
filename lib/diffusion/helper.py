import cv2
import numpy as np, open3d as o3d
import open3d.visualization.gui as gui
import secrets
import time, os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from geom.surfaces import bspline_surface_mesh_from_ctrl, project_external_along_normals_noreject,\
    cg_centeric_xy_spline
from geom.grids import infer_grid, reorder_ctrl_points_rowmajor
from utils.o3dviz import mat_points, mat_mesh, mat_mesh_tinted
import open3d.visualization.rendering as rendering


def _load_cloud_points(path_or_arr):
    if isinstance(path_or_arr, np.ndarray):
        pts = np.asarray(path_or_arr, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("cloud_pts ndarray must be (N,3)+")
        return pts[:, :3].copy()
    path = str(path_or_arr)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path)
        pts = arr["points"] if isinstance(arr, np.lib.npyio.NpzFile) and "points" in arr else np.array(arr)
        pts = np.asarray(pts, dtype=float)
    else:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Failed to read or empty point cloud: {path}")
        pts = np.asarray(pcd.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("Point cloud must be (N,3)+")
    return pts[:, :3].copy()

class RandomSurfacer:
    """
    - Builds base flat surface at z0 and its parallel top/bottom (z0±OF).
    - Precomputes Z_up/Z_down for each control point by vertical projection.
    - generate(N) -> list[np.ndarray]: N random control grids with fixed (x,y) and z ~ U[Z_down, Z_up].
    - draw(random_ctrl_pts): renders ext cloud, base/top/bottom AND a *smooth random B-spline surface* from random_ctrl_pts.
    """
    def __init__(self, cloud, grid_w=6, grid_h=4, samples_u=40, samples_v=40, margin=0.02, seed=None):
        # ---- robust, instance-scoped RNG (ignores global np.random.seed) ----
        np.random.seed(0)
        # Mix multiple entropy sources so two identical constructions in the same process still differ.
        entropy = [
            secrets.randbits(32),
            int(time.time_ns() & 0xFFFFFFFF),
            grid_w, grid_h, samples_u, samples_v
        ]
        ss = np.random.SeedSequence(entropy)
        self.rng = np.random.default_rng(ss)

        self.cloud_pts = _load_cloud_points(cloud)
        self.grid_w, self.grid_h = int(grid_w), int(grid_h)
        self.samples_u, self.samples_v = int(samples_u), int(samples_v)
        self.margin = float(margin)

        
        (self.base_mesh, self.ctrl_pts, self.W, self.H,
         self.center_xy, self.z0) = cg_centeric_xy_spline(
            self.cloud_pts, self.grid_w, self.grid_h, self.samples_u, self.samples_v, self.margin
        )

        z_mean = float(self.cloud_pts[:, 2].mean())
        z_min  = float(self.cloud_pts[:, 2].min())
        self.OF = z_mean - z_min

        # (Open3D note: .clone() is safest on newer versions)
        try:
            self.top_mesh = self.base_mesh.clone()
            self.bottom_mesh = self.base_mesh.clone()
        except AttributeError:
            self.top_mesh = o3d.geometry.TriangleMesh(self.base_mesh)
            self.bottom_mesh = o3d.geometry.TriangleMesh(self.base_mesh)

        self.top_mesh.translate((0, 0, +self.OF))
        self.bottom_mesh.translate((0, 0, -self.OF))

        self.Z_up_scalar   = self.z0 + self.OF
        self.Z_down_scalar = self.z0 - self.OF
        self.Z_up   = np.full((self.grid_h, self.grid_w), self.Z_up_scalar,   dtype=float)
        self.Z_down = np.full((self.grid_h, self.grid_w), self.Z_down_scalar, dtype=float)

        self.ctrl_xy = self.ctrl_pts[:, :2].copy()

    def generate(self, N: int) -> list[np.ndarray]:
        N = int(N)
        if N <= 0:
            return []
        z_lo = self.Z_down.ravel()
        z_hi = self.Z_up.ravel()
        outs = []
        for _ in range(N):
            # Use the instance RNG so draws differ across instances/runs even with same params
            z = self.rng.uniform(low=z_lo, high=z_hi, size=z_lo.shape).astype(float)
            outs.append(np.column_stack([self.ctrl_xy, z]))
        return outs

    def _mesh_from_ctrl(self, ctrl_pts_flat: np.ndarray) -> o3d.geometry.TriangleMesh:
        return bspline_surface_mesh_from_ctrl(
            ctrl_pts_flat, self.grid_w, self.grid_h, self.samples_u, self.samples_v
        )

    def draw(self, random_ctrl_pts: np.ndarray | None = None):
        ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.ctrl_pts))
        ctrl_pcd.paint_uniform_color([1.0, 0.6, 0.2])

        ext_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.cloud_pts))
        ext_pcd.paint_uniform_color([0.2, 0.6, 1.0])

        rnd_mesh = None
        if random_ctrl_pts is not None:
            rnd_mesh = self._mesh_from_ctrl(random_ctrl_pts)

        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("RandomSurfacer Viewer", 1280, 900)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)

        scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
        scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
        scene.scene.add_geometry("base_mesh", self.base_mesh, mat_mesh())
        scene.scene.add_geometry("top_mesh", self.top_mesh, mat_mesh_tinted((0.7, 0.95, 0.7, 0.55)))
        scene.scene.add_geometry("bottom_mesh", self.bottom_mesh, mat_mesh_tinted((0.95, 0.7, 0.7, 0.55)))
        if rnd_mesh is not None:
            rnd_mesh.compute_vertex_normals()
            scene.scene.add_geometry("random_spline", rnd_mesh, mat_mesh_tinted((0.8, 0.8, 0.2, 0.9)))

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        scene.scene.add_geometry("axis", axis, mat_mesh())

        # Fit camera to everything
        geoms = [ext_pcd, ctrl_pcd, self.base_mesh, self.top_mesh, self.bottom_mesh]
        if rnd_mesh is not None:
            geoms.append(rnd_mesh)
        aabbs = [g.get_axis_aligned_bounding_box() for g in geoms]
        mins = np.min([a.min_bound for a in aabbs], axis=0)
        maxs = np.max([a.max_bound for a in aabbs], axis=0)
        center = 0.5 * (mins + maxs)
        extent = float(np.max(maxs - mins))
        eye = center + np.array([0.0, -3.0 * extent, 1.8 * extent])
        up = np.array([0.0, 0.0, 1.0])
        scene.scene.camera.look_at(center, eye, up)

        gui.Application.instance.run()

def compute_shifted_ctrl_points(
    ext_pts: np.ndarray,
    ctrl_pts: np.ndarray,
    su: int = 100,
    sv: int = 100,
    k: float = 1.1):
    # --- Load control points ---
    ctrl_pts_orig = ctrl_pts.copy()
    if ctrl_pts.ndim == 1:
        ctrl_pts = ctrl_pts[None, :]
    if ctrl_pts.shape[1] < 3:
        raise RuntimeError("Control CSV must have at least 3 columns (x,y,z).")
    ctrl_pts = ctrl_pts[:, :3].astype(float)

    # --- Infer grid + reorder ---
    gw, gh = infer_grid(ctrl_pts)
    print(f"[compute_shifted_ctrl_points] Inferred control grid: gw={gw}, gh={gh}  (total={ctrl_pts.shape[0]})")
    ctrl_pts_rowmajor = reorder_ctrl_points_rowmajor(ctrl_pts)

    # --- Build initial spline mesh (used for projection) ---
    print(f"[compute_shifted_ctrl_points] Building B-spline surface mesh with su={su}, sv={sv} ...")
    mesh = bspline_surface_mesh_from_ctrl(ctrl_pts_rowmajor, gw, gh, su, sv)
    mesh.compute_vertex_normals()

    # --- Project external points along normals to the surface ---
    print(f"[compute_shifted_ctrl_points] Projecting {ext_pts.shape[0]} points along mesh normals (ray casting ±n)...")
    proj = project_external_along_normals_noreject(ext_pts, mesh)

    # --- Classify by world-Z and compute longest 'below' distance ---
    dz = ext_pts[:, 2] - proj[:, 2]
    eps = 1e-12
    below_mask = dz <= eps  # "below" or on the surface
    if not np.any(below_mask):
        print("[compute_shifted_ctrl_points] No 'below' points found. Z-displacement = 0.")
        disp = 0.0
    else:
        diffs_below = ext_pts[below_mask] - proj[below_mask]
        lengths_below = np.linalg.norm(diffs_below, axis=1)
        # numpy.max on empty would fail, but we've guarded with np.any
        disp = float(lengths_below.max(initial=0.0))
        print(f"[compute_shifted_ctrl_points] Computed Z-displacement (longest 'below' distance): {disp:.9f}")

    # --- Compute final mesh translate amount (K * disp) ---
    shft = float(k) * float(disp)
    print(f"[compute_shifted_ctrl_points] shift to apply on mesh (K*disp) = {shft:.9f}")

    # --- Prepare shifted control points (same behavior as original script: ctrl z -= disp) ---
    shifted_ctrl_pts = ctrl_pts_orig.astype(float)
    # Ensure we only modify the z-component if available
    if shifted_ctrl_pts.ndim == 1:
        # single control point row case
        if shifted_ctrl_pts.size >= 3:
            shifted_ctrl_pts[2] = shifted_ctrl_pts[2] - disp
    else:
        shifted_ctrl_pts[:, 2] = shifted_ctrl_pts[:, 2] - disp

    # Build shifted mesh by translating the original mesh by -shft in Z (same behavior as original)
    mesh_shifted = o3d.geometry.TriangleMesh(mesh)
    mesh_shifted.translate([0.0, 0.0, -shft], relative=True)
    mesh_shifted.compute_vertex_normals()

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.20, 0.70, 1.00])  # initial spline (cyan-ish)

    mesh_shifted.paint_uniform_color([0.90, 0.30, 0.40])  # shifted spline (distinct color)
    return shifted_ctrl_pts, mesh, shft, mesh_shifted

def create_grid_on_surface(mesh, samples_u, samples_v):
    # Get the mesh vertices and make a grid of points based on the surface
    vertices = np.asarray(np.asarray(mesh.vertices))
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)

    # Create a grid in the xy-plane based on the bounds of the mesh
    x_vals = np.linspace(min_bound[0], max_bound[0], samples_u)
    y_vals = np.linspace(min_bound[1], max_bound[1], samples_v)
    
    # Create a grid of x, y coordinates
    grid_points = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)

    # Now, project the grid onto the surface of the mesh by interpolating the z-values
    grid_points_3d = []
    for point in grid_points:
        # Simple nearest-neighbor approach to get the z-coordinate
        closest_vertex = None
        min_distance = float('inf')
        for vertex in vertices:
            dist = np.linalg.norm(vertex[:2] - point)
            if dist < min_distance:
                min_distance = dist
                closest_vertex = vertex
        grid_points_3d.append([point[0], point[1], closest_vertex[2]])

    grid_points_3d = np.array(grid_points_3d)

    # Create the point cloud for the grid points
    grid_pcd = o3d.geometry.PointCloud()
    grid_pcd.points = o3d.utility.Vector3dVector(grid_points_3d)
    return np.asarray(grid_pcd.points), grid_pcd

def score_based_downsample(ext_pts: np.ndarray,
                           scores01: np.ndarray,
                           target_fraction: float):
    """
    Probabilistic downsample based on exponential mapping of scores in [0,1],
    with a hard target fraction for maximum kept points.

    Parameters:
        ext_pts: (N,3) or (N,...) array of points (index aligned with scores01)
        scores01: (N,) scores in [0,1]
        seed: optional RNG seed
        target_fraction: fraction of points to keep at most (0 < target_fraction <= 1)

    Returns:
        ds_pts: np.ndarray of kept points
        ds_scores: np.ndarray of kept scores
        mask: boolean mask into original arrays (len == len(ext_pts))
    """
    scores = np.asarray(scores01, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores01 must be a 1-D array")
    N = scores.shape[0]
    if N == 0:
        return ext_pts[:0], scores[:0], np.zeros(0, dtype=bool)

    # clip scores
    scores = np.clip(scores, 0.0, 1.0)

    # exponential mapping: x -> (e^x - 1) / (e - 1)
    probs = (np.exp(scores) - 1.0) / (np.e - 1.0)

    rng = np.random.default_rng(None)

    # target number to keep
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be in (0, 1].")
    desired_k = max(1, int(np.ceil(N * target_fraction)))

    # If target >= N, just do Bernoulli sample using probs
    if desired_k >= N:
        draws = rng.random(size=probs.shape)
        mask = draws < probs
        # ensure at least one kept
        if not np.any(mask):
            mask[np.argmax(scores)] = True
    else:
        # If sum(probs) == 0 -> all zero probabilities: pick top-scoring desired_k points.
        total_prob = probs.sum()
        if total_prob <= 0.0:
            # choose highest-scoring indices
            idx = np.argsort(scores)[-desired_k:]
        else:
            # use probs as weights to choose exactly desired_k indices without replacement
            weights = probs / total_prob
            # rng.choice supports p parameter
            try:
                idx = rng.choice(N, size=desired_k, replace=False, p=weights)
            except Exception:
                # fallback: numeric edge-case; pick top by weighted random-permutation
                # (this branch rarely triggers)
                perm = rng.permutation(N)
                idx = perm[:desired_k]
        mask = np.zeros(N, dtype=bool)
        mask[np.asarray(idx)] = True

    ds_pts = ext_pts[mask]
    ds_scores = scores01[mask]
    return ds_pts, ds_scores, mask

def uniform_downsample(ext_pts: np.ndarray, target_fraction: float):
    """
    Uniformly downsample the given points to the specified fraction.

    Parameters:
        ext_pts: (N,3) or (N,...) array of points.
        target_fraction: fraction of points to keep (0 < target_fraction <= 1)

    Returns:
        ds_pts: np.ndarray of kept points
        mask: boolean mask into original array (len == len(ext_pts))
    """
    N = len(ext_pts)
    if N == 0:
        return ext_pts[:0], np.zeros(0, dtype=bool)

    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must be in (0, 1].")

    desired_k = max(1, int(np.ceil(N * target_fraction)))
    rng = np.random.default_rng(None)

    # Randomly choose desired_k unique indices
    idx = rng.choice(N, size=desired_k, replace=False)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True

    ds_pts = ext_pts[mask]
    return ds_pts, mask

def mask2image(bgmask, H, W):
    """
    Convert a flat boolean/numeric mask of length H*W into a binary image and save it.

    Parameters
    ----------
    bgmask : array-like
        1D mask of length H*W. Values will be treated as boolean (nonzero -> True).
    H, W : int
        Image height and width.
    out_path : str, optional
        Where to save the image (defaults to "bgmask.png").

    Returns
    -------
    img : np.ndarray
        The saved 2D uint8 image of shape (H, W) with values {0, 255}.
    """
    m = np.asarray(bgmask).ravel()
    if m.size != H * W:
        raise ValueError(f"bgmask has length {m.size}, but H*W = {H*W}.")

    # Ensure boolean semantics (True for foreground, for example)
    m_bool = m.astype(bool)
    m_inv = ~m_bool

    # Reshape row-major into (H, W)
    img = (m_inv.astype(np.uint8) * 255).reshape(H, W)

    return img
