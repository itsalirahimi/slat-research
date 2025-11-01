import numpy as np
import os, sys

from kinematics.transformations import rotation_matrix_x
from utils.conversion import pcm2pcd
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
from kinematics.pose import Pose
import open3d as o3d

from types import SimpleNamespace
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from scipy.spatial import cKDTree
_HAS_SCIPY = True
import cv2

def add_grid_and_axes(extent=50, step=1):
    """Return a grey XY-grid and XYZ axes for context."""
    pts, lines = [], []
    for y in np.arange(-extent, extent+1e-6, step):
        pts += [[-extent, y, 0], [ extent, y, 0]]
        lines.append([len(pts)-2, len(pts)-1])
    for x in np.arange(-extent, extent+1e-6, step):
        pts += [[x, -extent, 0], [ x, extent, 0]]
        lines.append([len(pts)-2, len(pts)-1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )
    grid.colors = o3d.utility.Vector3dVector([[0.2,0.2,0.2]] * len(lines))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    return grid, axes

def create_normal_lines(pcd, voxel_size=0.02, normal_length=1.0):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )
    pts = np.asarray(pcd_down.points)
    nrm = np.asarray(pcd_down.normals)
    seg_pts, seg_lines = [], []
    for i, (p, v) in enumerate(zip(pts, nrm)):
        seg_pts.append(p)
        seg_pts.append(p + v * normal_length)
        seg_lines.append([2*i, 2*i+1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(seg_pts),
        lines=o3d.utility.Vector2iVector(seg_lines)
    )
    ls.colors = o3d.utility.Vector3dVector([[0,0,1]] * len(seg_lines))
    return ls

def print_report(query_pts, best_in_idxs, best_out_idxs,
                 all_normals, all_dists, all_projs):
    H, W, k, _ = all_normals.shape
    sep = "-" * 60
    print("\nProjection Report:\n" + sep)
    for i in range(H):
        for j in range(W):
            bi, bo = best_in_idxs[i,j], best_out_idxs[i,j]
            q = query_pts[i,j]
            print(f"Query[{i},{j}] = {q.tolist()}")
            if bi >= 0:
                n = all_normals[i,j,bi]
                d = all_dists[i,j,bi]
                p = all_projs[i,j,bi]
                print(f"  Inside  → idx={bi:2d}, dist={d:7.3f}, "
                      f"normal={n.round(3).tolist()}, proj={p.round(3).tolist()}")
            else:
                print("  Inside  → none")
            if bo >= 0:
                n = all_normals[i,j,bo]
                d = all_dists[i,j,bo]
                p = all_projs[i,j,bo]
                print(f"  Outside → idx={bo:2d}, dist={d:7.3f}, "
                      f"normal={n.round(3).tolist()}, proj={p.round(3).tolist()}")
            else:
                print("  Outside → none")
        print(sep)

def _normalize_rows(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def project_points_multi_fast(
    pcd,
    query_pts,
    k=10,
    just_vals=False,
    get_norm=False,
    batch_size=50_000,
    dtype=np.float32,
    print_log=False,
    visualize=False,
    just_proj = False
):
    """
    Fast batched variant with optional logging and visualization.

    Returns:
      - if just_vals: (H, W, 1) float32 of signed distances
      - if get_norm:  (H, W, 3) float32 normal-distance vectors
      - else:         (H, W, 1) object array of dicts (matching original)
    """
    if not pcd.has_normals():
        raise ValueError("Estimate normals on the point cloud first.")

    # Pull data once, normalize, cast to float32 for bandwidth
    pts   = np.asarray(pcd.points, dtype=dtype)
    norms = _normalize_rows(np.asarray(pcd.normals, dtype=dtype))

    n_pts = pts.shape[0]
    k_eff = int(min(k, n_pts))

    # Bounds for inside/outside classification (XY box)
    pts_xy = pts[:, :2]
    min_x, max_x = pts_xy[:, 0].min(), pts_xy[:, 0].max()
    min_y, max_y = pts_xy[:, 1].min(), pts_xy[:, 1].max()

    H, W, _ = query_pts.shape
    Q = H * W
    q_all = np.asarray(query_pts, dtype=dtype).reshape(Q, 3)

    # Build KD-tree once
    if not _HAS_SCIPY:
        raise ImportError(
            "SciPy is required for the fast batched KD-tree path. "
            "Install it or switch to sklearn NearestNeighbors."
        )
    tree = cKDTree(pts)

    # Output flat buffers (later reshaped)
    out_dist = np.full((Q,), np.nan, dtype=dtype)
    out_nvec = np.full((Q, 3), np.nan, dtype=dtype)
    out_proj = np.full((Q, 3), np.nan, dtype=dtype)
    out_idx  = np.full((Q,), -1, dtype=np.int32)

    # Optional per-k caches (only if logging or viz)
    keep_candidates = (print_log or visualize)
    if keep_candidates:
        all_normals_flat = np.zeros((Q, k_eff, 3), dtype=dtype)
        all_dists_flat   = np.zeros((Q, k_eff),    dtype=dtype)
        all_projs_flat   = np.zeros((Q, k_eff, 3), dtype=dtype)
        best_in_flat     = np.full((Q,), -1, dtype=np.int32)
        best_out_flat    = np.full((Q,), -1, dtype=np.int32)

    # Process queries in batches to limit memory
    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        qb = q_all[start:end]  # (B,3)
        B = qb.shape[0]

        # KNN indices for the whole batch
        _, idxs = tree.query(qb, k=k_eff, workers=-1)
        idxs = np.asarray(idxs, dtype=np.intp)
        if idxs.ndim == 1:  # ensure (B, k)
            idxs = idxs[:, None]

        # Gather neighbor points and normals
        neigh_pts = pts[idxs]      # (B,k,3)
        neigh_n   = norms[idxs]    # (B,k,3)

        # Signed distances and foot points
        vecs   = qb[:, None, :] - neigh_pts                  # (B,k,3)
        dists  = np.einsum("bki,bki->bk", vecs, neigh_n)     # (B,k)
        projs  = qb[:, None, :] - dists[..., None] * neigh_n # (B,k,3)

        # Inside/outside mask by XY bounds
        px = projs[..., 0]
        py = projs[..., 1]
        in_mask  = (px >= min_x) & (px <= max_x) & (py >= min_y) & (py <= max_y)
        out_mask = ~in_mask

        # Best inside / outside by smallest |distance|
        absd = np.abs(dists)

        absd_in = np.where(in_mask, absd, np.inf)
        bi = np.argmin(absd_in, axis=1)          # (B,)
        best_in = absd_in[np.arange(B), bi]
        has_in = np.isfinite(best_in)
        d_in = dists[np.arange(B), bi]
        d_in[~has_in] = np.inf

        absd_out = np.where(out_mask, absd, np.inf)
        bo = np.argmin(absd_out, axis=1)
        best_out = absd_out[np.arange(B), bo]
        has_out = np.isfinite(best_out)
        d_out = dists[np.arange(B), bo]
        d_out[~has_out] = np.inf

        # Final selection between inside/outside
        choose_in = (np.abs(d_in) <= np.abs(d_out))
        sel_idx   = np.where(choose_in, bi, bo)         # (B,)
        sel_dist  = np.where(choose_in, d_in, d_out)    # (B,)

        valid = np.isfinite(sel_dist)
        sel_idx_final  = np.where(valid, sel_idx, -1)
        sel_dist_final = np.where(valid, sel_dist, np.nan)

        # Write outputs
        out_dist[start:end] = sel_dist_final
        out_idx[start:end]  = sel_idx_final

        if valid.any():
            vi = np.where(valid)[0]
            kk = sel_idx_final[vi]

            n_sel = neigh_n[vi, kk]            # (V,3)
            p_sel = projs[vi, kk]              # (V,3)
            d_sel = sel_dist_final[vi][:, None]# (V,1)

            out_nvec[start:end][vi] = n_sel * d_sel
            out_proj[start:end][vi] = p_sel

        # Keep candidate fields if requested
        if keep_candidates:
            all_normals_flat[start:end] = neigh_n
            all_dists_flat[start:end]   = dists
            all_projs_flat[start:end]   = projs
            best_in_flat[start:end]     = bi
            best_out_flat[start:end]    = bo

    # Quick-return formats
    if just_vals:
        return out_dist.reshape(H, W, 1).astype(np.float32)

    if get_norm:
        return out_nvec.reshape(H, W, 3).astype(np.float32)
    
    if just_proj:
        return out_proj.reshape(H, W, 3).astype(np.float32)

    # Optional log and visualization
    if keep_candidates:
        all_normals = all_normals_flat.reshape(H, W, k_eff, 3)
        all_dists   = all_dists_flat.reshape(H, W, k_eff)
        all_projs   = all_projs_flat.reshape(H, W, k_eff, 3)
        best_in_idxs  = best_in_flat.reshape(H, W)
        best_out_idxs = best_out_flat.reshape(H, W)

        # Print log if requested
        if print_log:
            print_report(
                query_pts,
                best_in_idxs, best_out_idxs,
                all_normals, all_dists, all_projs
            )

        # Visualize if requested
        if visualize:
            # Prepare flattened arrays (float64 for Open3D)
            N      = H * W
            q_flat = query_pts.reshape(N, 3).astype(np.float64)
            pk     = all_projs.reshape(N, k_eff, 3).astype(np.float64)

            # Purple: all candidate lines
            q_rep     = np.repeat(q_flat, k_eff, axis=0)
            p_rep     = pk.reshape(N * k_eff, 3)
            pts_all   = np.vstack([q_rep, p_rep])
            lines_all = [[i, i + N * k_eff] for i in range(N * k_eff)]
            ls_all = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts_all),
                lines=o3d.utility.Vector2iVector(lines_all)
            )
            ls_all.colors = o3d.utility.Vector3dVector([[0.5, 0, 0.5]] * len(lines_all))

            # Green/red: best inside/outside
            pts_in, lines_in = [], []
            pts_out, lines_out = [], []
            bi_flat = best_in_idxs.flatten()
            bo_flat = best_out_idxs.flatten()
            for idx in range(N):
                bi = bi_flat[idx]
                if bi >= 0:
                    s, e = q_flat[idx], pk[idx, bi]
                    b = len(pts_in)
                    pts_in += [s, e]
                    lines_in.append([b, b + 1])
                bo = bo_flat[idx]
                if bo >= 0:
                    s, e = q_flat[idx], pk[idx, bo]
                    b = len(pts_out)
                    pts_out += [s, e]
                    lines_out.append([b, b + 1])

            ls_in = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.asarray(pts_in, dtype=np.float64)),
                lines=o3d.utility.Vector2iVector(lines_in)
            )
            ls_in.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines_in))

            ls_out = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(np.asarray(pts_out, dtype=np.float64)),
                lines=o3d.utility.Vector2iVector(lines_out)
            )
            ls_out.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines_out))

            # Points
            qp = o3d.geometry.PointCloud()
            qp.points = o3d.utility.Vector3dVector(q_flat)
            qp.paint_uniform_color([0, 1, 0])

            pp_in  = [pk[i, bi_flat[i]] for i in range(N) if bi_flat[i] >= 0]
            print(pp_in)
            pp_out = [pk[i, bo_flat[i]] for i in range(N) if bo_flat[i] >= 0]
            pc_in  = o3d.geometry.PointCloud()
            pc_in.points = o3d.utility.Vector3dVector(np.asarray(pp_in, dtype=np.float64))
            pc_in.paint_uniform_color([0, 0, 1])
            pc_out = o3d.geometry.PointCloud()
            pc_out.points = o3d.utility.Vector3dVector(np.asarray(pp_out, dtype=np.float64))
            pc_out.paint_uniform_color([1, 0, 0])

            # Grid, axes, downsampled normals (helpers must be defined)
            grid, axes   = add_grid_and_axes()
            normals_ls   = create_normal_lines(pcd)

            # o3d.visualization.draw_geometries([
            #     pcd, qp, pc_in, pc_out,
            #     ls_all, ls_in, ls_out,
            #     grid, axes, normals_ls
            # ])
            o3d.visualization.draw_geometries([
                pcd, qp, pc_in, pc_out,
                ls_all, ls_in, ls_out,
                
                
            ])

    # Full nested dict object array (note: building 200k dicts is inherently heavy)
    full = np.empty((H, W, 1), dtype=object)
    flat_idx = 0
    for i in range(H):
        for j in range(W):
            di = out_dist[flat_idx]
            ni = out_nvec[flat_idx]
            pi = out_proj[flat_idx]
            ii = int(out_idx[flat_idx])
            full[i, j, 0] = {
                "query_point":            q_all[flat_idx],
                "projected_index":        ii if ii >= 0 else None,
                "projected_point":        None if np.isnan(pi).any() else pi,
                "normal_distance_vector": None if np.isnan(ni).any() else ni,
                "distance":               None if np.isnan(di) else float(di),
            }
            flat_idx += 1
    return full

def euclidean_distance_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel Euclidean distance between two H×W×3 arrays.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays of shape (H, W, 3). Types can be integer or float.

    Returns
    -------
    np.ndarray
        Distance map of shape (H, W, 1), dtype float32.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a{a.shape} vs b{b.shape}")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {a.shape}")

    # Cast once to float for precise diff; keepdims=True → (H, W, 1)
    diff = a.astype(np.float32) - b.astype(np.float32)
    dist = np.linalg.norm(diff, axis=2, keepdims=True)  # (H, W, 1)
    return dist

# -------------------- border index helpers --------------------
def border_grid_indices(shape):
    """
    Given image shape (h, w), return a (N, 2) array of (row, col) integer
    pixel indices that lie ONLY on the image borders (corners included).
    Spacing along each axis is as equal as possible with integer pixels.
    """
    h, w = int(shape[0]), int(shape[1])
    assert h >= 2 and w >= 2, "shape must be at least 2x2"

    def _evenly_spaced_indices(n: int, max_index: int) -> np.ndarray:
        # produce n integers from 0..max_index with near-uniform steps
        assert n >= 2
        intervals = n - 1
        base = max_index // intervals
        rem = max_index % intervals
        out = [0]
        pos = 0
        err = 0
        for _ in range(intervals):
            step = base
            err += rem
            if err >= intervals:
                step += 1
                err -= intervals
            pos += step
            out.append(pos)
        out[-1] = max_index
        return np.asarray(out, dtype=np.int32)

    target_step = max(10, min(h, w) // 6)
    nx = max(2, (w - 1) // target_step + 1)
    ny = max(2, (h - 1) // target_step + 1)

    xs = _evenly_spaced_indices(nx, w - 1)
    ys = _evenly_spaced_indices(ny, h - 1)

    # Assemble borders: top, bottom, left, right
    top    = np.stack([np.zeros_like(xs),       xs], axis=1)                 # (0, x)
    bottom = np.stack([np.full_like(xs, h - 1), xs], axis=1)                 # (h-1, x)
    left   = np.stack([ys,                      np.zeros_like(ys)], axis=1)  # (y, 0)
    right  = np.stack([ys,                      np.full_like(ys, w - 1)], axis=1)  # (y, w-1)

    rc = np.concatenate([top, bottom, left, right], axis=0).astype(np.int32)

    # Deduplicate corners while preserving first occurrences
    keys = rc[:, 0] * w + rc[:, 1]
    _, uniq_idx = np.unique(keys, return_index=True)
    rc = rc[np.sort(uniq_idx)]
    return rc


def border_all_indices(shape):
    """
    Dense border indices for 'every single' border pixel.
    Returns dict with 'top','bottom','left','right' -> array[(N,2)] of (i,j).
    Corners will be present in both touching sides; users can dedup if needed.
    """
    h, w = int(shape[0]), int(shape[1])
    assert h >= 2 and w >= 2, "shape must be at least 2x2"
    top    = np.stack([np.zeros(w, dtype=np.int32),              np.arange(w, dtype=np.int32)], axis=1)
    bottom = np.stack([np.full(w, h - 1, dtype=np.int32),        np.arange(w, dtype=np.int32)], axis=1)
    left   = np.stack([np.arange(h, dtype=np.int32),             np.zeros(h, dtype=np.int32)], axis=1)
    right  = np.stack([np.arange(h, dtype=np.int32),             np.full(h, w - 1, dtype=np.int32)], axis=1)
    return dict(top=top, bottom=bottom, left=left, right=right)


# -------------------- point-cloud helpers (unchanged) --------------------
def finite_points_from_pcd(pcd):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError("Point cloud has no points.")
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.size == 0:
        raise ValueError("All points are non-finite (NaN/Inf).")
    return pts

def compute_center_of_mass(pts):
    return pts.mean(axis=0)

def build_kdtree_from_points(pts):
    # Fast SciPy KD-tree (balanced + compact)
    return cKDTree(pts, leafsize=32, compact_nodes=True, balanced_tree=True)

def nearest_index(kdtree, q, k=1):
    # cKDTree.query returns (dist, idx)
    _, idx = kdtree.query(q, k=k)
    return int(idx) if k == 1 else idx


def fit_local_plane(patch):
    c = patch.mean(axis=0)
    _, _, Vt = np.linalg.svd(patch - c, full_matrices=False)
    n = Vt[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, c

def ray_plane_intersection(origin, dir_vec, plane_pt, plane_n, forward_only=True):
    dirn = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    denom = float(np.dot(plane_n, dirn))
    if abs(denom) < 1e-12:
        return None
    t = float(np.dot(plane_n, plane_pt - origin) / denom)
    if forward_only and t <= 0:
        return None
    return origin + t * dirn

def make_sphere(center, radius, color):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    m.translate(center)
    return m

def make_lineset(points_np, lines_idx, color_rgb):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_np)
    ls.lines = o3d.utility.Vector2iVector(lines_idx)
    ls.colors = o3d.utility.Vector3dVector([color_rgb for _ in range(len(lines_idx))])
    return ls

def make_polyline(points_np, color_rgb):
    if points_np.shape[0] < 2:
        return None
    lines = np.column_stack([
        np.arange(points_np.shape[0]-1, dtype=np.int32),
        np.arange(1, points_np.shape[0], dtype=np.int32)
    ])
    return make_lineset(points_np, lines, color_rgb)

def sample_segment_interior(a, b, n):
    if n <= 0:
        return np.empty((0, 3), dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ts = (np.arange(n, dtype=float) + 1.0) / (n + 1.0)
    return a[None, :] + ts[:, None] * (b - a)[None, :]

def orthogonal_projection_to_local_surface(M, pts, kdtree, k_neighbors):
    k_use = min(max(3, k_neighbors), pts.shape[0])
    # query k nearest neighbors of M
    _, nn_idx = kdtree.query(M, k=k_use)
    patch = pts[np.atleast_1d(nn_idx)]
    n, c = fit_local_plane(patch)
    d = float(np.dot(M - c, n))
    P = M - d * n
    return P

def chord_length_param(points):
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.zeros(points.shape[0], dtype=float)
    t[1:] = np.cumsum(np.sqrt(diffs + 1e-12))
    if t[-1] > 0:
        t /= t[-1]
    return t

def cubic_spline_interp(points, n_samples=200):
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 2:
        return P.copy()
    if P.shape[0] == 2:
        t = np.linspace(0, 1, n_samples)
        return (1 - t)[:, None] * P[0] + t[:, None] * P[1]
    t = chord_length_param(P)
    ts = np.linspace(0, 1, n_samples)
    if _HAS_SCIPY  and P.shape[0] >= 3:
        xs = CubicSpline(t, P[:, 0], bc_type="natural")(ts)
        ys = CubicSpline(t, P[:, 1], bc_type="natural")(ts)
        zs = CubicSpline(t, P[:, 2], bc_type="natural")(ts)
        return np.stack([xs, ys, zs], axis=1)
    return catmull_rom_sample(P, t, ts)

def catmull_rom_sample(P, t, ts):
    Q = P
    tq = t
    seg_idx = np.searchsorted(tq, ts, side='right') - 1
    seg_idx = np.clip(seg_idx, 0, len(tq) - 2)
    def get(idx):
        idx = int(np.clip(idx, 0, len(Q)-1))
        return Q[idx]
    samples = []
    for s, u in zip(seg_idx, ts):
        i0, i1, i2, i3 = s-1, s, s+1, s+2
        P0, P1, P2, P3 = get(i0), get(i1), get(i2), get(i3)
        t0, t1, t2, t3 = tq[max(i0,0)], tq[i1], tq[i2], tq[min(i3, len(tq)-1)]
        if t2 - t1 < 1e-12:
            tau = 0.0
        else:
            tau = (u - t1) / (t2 - t1)
            tau = np.clip(tau, 0.0, 1.0)
        def alpha_tan(Pm, P0, P1, tm, t0, t1):
            denom = (t1 - tm) if (t1 - tm) > 1e-12 else 1e-12
            return (P1 - Pm) / denom
        m1 = ( (t2 - t1) * (alpha_tan(P0, P1, P2, t0, t1, t2)) +
               (t1 - t0) * (alpha_tan(P1, P2, P3, t1, t2, t3)) ) * 0.5
        m2 = ( (t3 - t2) * (alpha_tan(P1, P2, P3, t1, t2, t3)) +
               (t2 - t1) * (alpha_tan(P0, P1, P2, t0, t1, t2)) ) * 0.5
        tt, tt2, tt3 = tau, tau*tau, tau*tau*tau
        h00, h10, h01, h11 = 2*tt3-3*tt2+1, tt3-2*tt2+tt, -2*tt3+3*tt2, tt3-tt2
        C = (h00*P1 + h10*m1 + h01*P2 + h11*m2)
        samples.append(C)
    return np.asarray(samples, dtype=float)

def _skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

def rot_from_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); s = np.linalg.norm(v); c = np.dot(a, b)
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, axis); v = v / (np.linalg.norm(v) + 1e-12)
        K = _skew(v); return np.eye(3) + K + K @ K
    K = _skew(v)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s**2 + 1e-12))

def make_arrow_from_to(start, end, color, thin_factor=1.0):
    start = np.asarray(start, float); end = np.asarray(end, float)
    v = end - start; L = float(np.linalg.norm(v))
    if L < 1e-12:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=1e-6)
        m.paint_uniform_color(color); m.translate(start); return m
    cone_len = 0.20 * L; cyl_len = max(L - cone_len, 1e-9)
    shaft_r = 0.02 * L * thin_factor; head_r = 2.0 * shaft_r
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_r, cone_radius=head_r,
        cylinder_height=cyl_len, cone_height=max(L-cyl_len, 1e-9))
    arrow.compute_vertex_normals(); arrow.paint_uniform_color(color)
    R = rot_from_a_to_b(np.array([0.0, 0.0, 1.0]), v)
    arrow.rotate(R, center=np.array([0.0, 0.0, 0.0])); arrow.translate(start)
    return arrow


# -------------------- runtime config (UNCHANGED core + vis_debug) --------------------
CONFIG = SimpleNamespace(
    k_plane=50,
    k_proj=40,
    allow_backward=False,
    n_mid=10,
    spline_samples=200,
    spline_color=[1.0, 1.0, 0.0],  # yellow for unfold-spline
    show_green_to_orange=True,
    n_greens=10,
    vis_debug=True,
)


# -------------------- core functions (UNCHANGED math) --------------------
def find_surface_mass_anchor(surface_pts):
    cg = compute_center_of_mass(surface_pts)
    kdt_all = build_kdtree_from_points(surface_pts)
    ni = nearest_index(kdt_all, cg)
    k_use = min(max(3, CONFIG.k_plane), surface_pts.shape[0])

    # cKDTree: get neighbor indices around the nearest surface point
    _, nn_idx = kdt_all.query(surface_pts[ni], k=k_use)
    patch = surface_pts[np.atleast_1d(nn_idx)]

    n, c_on_plane = fit_local_plane(patch)
    if np.dot(n, c_on_plane - cg) < 0.0:
        n = -n
    surface_interior_cg = ray_plane_intersection(
        cg, n, c_on_plane, n, forward_only=(not CONFIG.allow_backward)
    )
    if surface_interior_cg is None or not np.all(np.isfinite(surface_interior_cg)):
        raise SystemExit("Could not compute 'surface-interior-cg'. Adjust parameters.")
    aux = dict(cg=cg, kdtree=kdt_all, nn_patch_idx=np.atleast_1d(nn_idx),
               plane_normal=n, plane_point=c_on_plane)
    return surface_interior_cg, aux

def find_surface_point_unfold_params(surface_pts, surface_interior_cg, point_i):
    mids = sample_segment_interior(surface_interior_cg, point_i, CONFIG.n_mid)
    kdt_all = build_kdtree_from_points(surface_pts)
    proj_points = [orthogonal_projection_to_local_surface(M, surface_pts, kdt_all, CONFIG.k_proj) for M in mids]
    ctrl_pts_np = np.asarray([surface_interior_cg, *proj_points, point_i], dtype=float)
    curve = cubic_spline_interp(ctrl_pts_np, n_samples=CONFIG.spline_samples)

    if curve.shape[0] >= 2:
        v = curve[1] - curve[0]
        dir_vec = v / (np.linalg.norm(v) + 1e-12)
    else:
        v = point_i - surface_interior_cg
        dir_vec = v / (np.linalg.norm(v) + 1e-12)

    diffs = np.diff(curve, axis=0) if curve.shape[0] >= 2 else np.empty((0,3))
    seglens = np.linalg.norm(diffs, axis=1) if diffs.size else np.array([0.0])
    dist = float(np.sum(seglens))

    aux = dict(mids=mids, proj_points=np.asarray(proj_points, dtype=float), curve=curve)
    return dir_vec, dist, aux

def unfold_surface_point(surface_interior_cg, dir_vec, dist):
    # move in plane G along projected cyan dir
    dir_xy = np.array([dir_vec[0], dir_vec[1], 0.0], dtype=float)
    nrm = np.linalg.norm(dir_xy)
    if nrm < 1e-12:
        dir_xy = np.array([1.0, 0.0, 0.0], dtype=float); nrm = 1.0
    u = dir_xy / nrm
    p = np.asarray(surface_interior_cg, dtype=float) + dist * u
    p[2] = surface_interior_cg[2]
    return p


# -------------------- visualization (existing) --------------------
def _make_dashed_line(a, b, color_rgb, dash_len, gap_len):
    a = np.asarray(a, float); b = np.asarray(b, float)
    v = b - a; L = float(np.linalg.norm(v))
    if L < 1e-12: return None
    u = v / L; pts = [a]; lines = []; pos = 0.0; draw_on = True; idx = 0
    while pos < L - 1e-12:
        step = dash_len if draw_on else gap_len
        step = min(step, L - pos)
        next_pt = a + (pos + step) * u
        pts.append(next_pt)
        if draw_on: lines.append([idx, idx + 1])
        idx += 1; pos += step; draw_on = not draw_on
    return make_lineset(np.vstack(pts), np.asarray(lines, np.int32), color_rgb)

def _make_plane_G(bbox, z_val, color=[0.2, 0.6, 1.0]):
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    corners = np.array([
        [min_bound[0], min_bound[1], z_val],
        [max_bound[0], min_bound[1], z_val],
        [max_bound[0], max_bound[1], z_val],
        [min_bound[0], max_bound[1], z_val],
    ], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def visualize_scene_multi(pcd, cg, surface_interior_cg, items, spline_color, show_green_to_orange=True):
    if not CONFIG.vis_debug:
        return  # skip any visualization work entirely

    draw = [pcd]
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent()) if bbox is not None else 1.0

    plane_G = _make_plane_G(bbox, surface_interior_cg[2], color=[0.2, 0.6, 1.0])
    draw.append(plane_G)

    r_cg = max(1e-6, 0.0020 * diag)
    r_sicg = max(1e-6, 0.0018 * diag)
    draw.append(make_sphere(cg, r_cg, [1.0, 0.0, 0.0]))
    draw.append(make_sphere(surface_interior_cg, r_sicg, [0.0, 0.0, 1.0]))

    r_point = max(1e-6, 0.0018 * diag)
    r_mid_purple = max(1e-6, 0.0015 * diag)
    r_mid_yellow = max(1e-6, 0.0015 * diag)
    r_unfolded = max(1e-6, 0.0018 * diag)
    r_radial_proj = max(1e-6, 0.0018 * diag)

    dash_len = 0.02 * diag
    gap_len  = 0.015 * diag

    for idx, it in enumerate(items, 1):
        point_i = it["point"]
        mids_purple = it.get("mids", np.empty((0,3)))
        mids_yellow = it.get("proj_points", np.empty((0,3)))
        curve = it.get("curve", None)
        unfolded_i = it.get("unfolded", None)
        radial_proj_i = it.get("radial_proj", None)

        draw.append(make_sphere(point_i, r_point, [0.0, 1.0, 0.0]))
        draw.append(make_lineset(np.vstack([surface_interior_cg, point_i]),
                                 np.array([[0, 1]], np.int32), [1.0, 1.0, 1.0]))
        for M in mids_purple:
            draw.append(make_sphere(M, r_mid_purple, [1.0, 0.0, 1.0]))
        for P in mids_yellow:
            draw.append(make_sphere(P, r_mid_yellow, [1.0, 1.0, 0.0]))

        if curve is not None and curve.shape[0] >= 2:
            ls = make_polyline(curve, CONFIG.spline_color)
            if ls is not None: draw.append(ls)
            p0, p1 = curve[0], curve[1]
            draw.append(make_arrow_from_to(p0, p1 + 19*(p1 - p0), [0.0, 1.0, 1.0], thin_factor=0.5))

        if unfolded_i is not None:
            draw.append(make_sphere(unfolded_i, r_unfolded, [1.0, 0.5, 0.0]))
            if show_green_to_orange:
                ds = _make_dashed_line(point_i, unfolded_i, [1.0, 0.5, 0.0], dash_len, gap_len)
                if ds is not None: draw.append(ds)

        if radial_proj_i is not None:
            draw.append(make_sphere(radial_proj_i, r_radial_proj, [0.6, 0.6, 0.6]))
            if show_green_to_orange:
                ds_g = _make_dashed_line(point_i, radial_proj_i, [0.6, 0.6, 0.6], dash_len, gap_len)
                if ds_g is not None: draw.append(ds_g)

    o3d.visualization.draw_geometries(
        draw,
        window_name="cg & surface-interior-cg with N×{point-(id), radial-dir-(id), midpoints, unfold-spline-(id), unfolded-/radial-projection-point-(id)} + plane G",
        width=1400, height=900
    )


# -------------------- new: spline evaluation utility --------------------
def eval_spline_through_points(points, ts_query):
    """
    Fit a C^2 cubic (SciPy) or Catmull-Rom fallback through 'points' and evaluate at ts_query in [0,1].
    """
    P = np.asarray(points, dtype=float)
    if P.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    if P.shape[0] == 1:
        return np.repeat(P, repeats=len(ts_query), axis=0)
    t_knots = chord_length_param(P)
    ts_query = np.asarray(ts_query, dtype=float)
    ts_query = np.clip(ts_query, 0.0, 1.0)
    if _HAS_SCIPY  and P.shape[0] >= 3:
        xs = CubicSpline(t_knots, P[:, 0], bc_type="natural")(ts_query)
        ys = CubicSpline(t_knots, P[:, 1], bc_type="natural")(ts_query)
        zs = CubicSpline(t_knots, P[:, 2], bc_type="natural")(ts_query)
        return np.stack([xs, ys, zs], axis=1)
    # fallback to Catmull-Rom eval at custom ts
    return catmull_rom_sample(P, t_knots, ts_query)

# -------------------- old high-level (kept for compatibility) --------------------
def unfold_image_border_points(
    xyz_img,
    hfov_deg: float = 66.0,
    pose_kwargs=None,
    max_points=None,
):
    """
    OLD high-level: read metric depth CSV, select uniform border pixels (i,j),
    run the EXISTING unfolding pipeline, and return the orange (unfolded) points.

    Returns
    -------
    unfolded_points : (N, 3) float32
    extras : dict with pcd/cg/surface_interior_cg/items
    """
    H, W = xyz_img.shape[:2]

    # 3) Flatten to pcd (black)
    pts_flat = xyz_img.reshape(-1, 3)
    np.savetxt("all.csv", xyz_img[:,:,0], delimiter=',')
    mask = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[mask]
    if pts_flat.size == 0:
        raise SystemExit("All projected points are non-finite.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_flat)
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Finite pts for math (unchanged)
    pts = finite_points_from_pcd(pcd)
    if pts.shape[0] != len(pcd.points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Step 1: surface-interior-cg via cg + local plane hit (unchanged)
    surface_interior_cg, aux1 = find_surface_mass_anchor(pts)
    cg = aux1["cg"]

    # Step 2: choose border (i,j) from HxW
    rc_all = border_grid_indices((H, W))
    if max_points is not None and max_points > 0:
        rc_use = rc_all[:max_points]
    else:
        rc_use = rc_all

    # Convert (i,j) -> 3D points, skip non-finite & duplicates of SICG
    points = []
    for (i, j) in rc_use:
        p_ij = xyz_img[i, j]
        if not np.all(np.isfinite(p_ij)):
            continue
        if np.linalg.norm(p_ij - surface_interior_cg) <= 1e-12:
            continue
        points.append(p_ij)
    if len(points) == 0:
        raise SystemExit("[ERROR] No valid border points found.")

    # Step 3: loop over points (UNCHANGED math)
    items = []
    unfolded_pts = []
    for idx, p in enumerate(points, 1):
        dir_vec, spline_len, aux2 = find_surface_point_unfold_params(pts, surface_interior_cg, p)
        white_len = float(np.linalg.norm(p - surface_interior_cg))
        unfolded_i = unfold_surface_point(surface_interior_cg, dir_vec, spline_len)
        radial_proj_i = unfold_surface_point(surface_interior_cg, dir_vec, white_len)
        unfolded_pts.append(unfolded_i)

        # Per-point printing only if debugging
        if CONFIG.vis_debug:
            print(f"[{idx:02d}] point-{idx}:  x={p[0]:.6f}, y={p[1]:.6f}, z={p[2]:.6f}")
            print(f"     unfolded-point-{idx}:  x={unfolded_i[0]:.6f}, y={unfolded_i[1]:.6f}, z={unfolded_i[2]:.6f}")
            print(f"     cyan-dir: ({dir_vec[0]:.6f}, {dir_vec[1]:.6f}, {dir_vec[2]:.6f})")
            print(f"     unfold-spline-{idx} length = {spline_len:.6f}")
            print(f"     radial-dir-{idx} length    = {white_len:.6f}")

        if CONFIG.vis_debug:
            items.append(dict(
                point=p,
                mids=aux2["mids"],
                proj_points=aux2["proj_points"],
                curve=aux2["curve"],
                unfolded=unfolded_i,
                radial_proj=radial_proj_i,
            ))

    extras = dict(
        pcd=pcd,
        cg=cg,
        surface_interior_cg=surface_interior_cg,
        items=items,
        xyz_img=xyz_img,  # expose for reuse
    )
    return np.asarray(unfolded_pts, dtype=np.float32), extras


def unfold_image_borders(
    xyz_img,
    hfov_deg: float = 66.0,
    pose_kwargs=None,
    max_points=None,
):
    """
    Superset of unfold_image_border_points:
      1) Compute orange (unfolded) points for a UNIFORM subset of border pixels.
      2) Build four splines that each go *corner→corner*, in this exact order:
         TOP:    top-left  -> top-right
         RIGHT:  top-right -> bottom-right
         BOTTOM: bottom-right -> bottom-left
         LEFT:   bottom-left -> top-left
         (each passes all mid-orange points on that side, if any)
      3) For EVERY border pixel, suggest an orange position by evaluating the
         respective side spline at that pixel’s normalized coordinate t∈[0,1].
      4) Returns the sampled oranges and the dense (tiny red) suggestions.

    Returns
    -------
    result : dict with:
      - 'unfolded_sample': (Ns,3) orange points for the sampled border set
      - 'splines': dict side -> {'P': (Nk,3) sample oranges incl. corners,
                                 'ts_dense': (Nd,), 'Y_dense': (Nd,3)}
      - 'dense_red': dict side -> (Nd,3) suggested orange positions (tiny red dots)
    extras : dict from unfold_image_border_points (pcd, cg, sicg, items, xyz_img, ...)
    """

    # --- 0) run the existing pipeline to get sample oranges + scene context ---
    unfolded_sample, extras = unfold_image_border_points(
        xyz_img=xyz_img,
        hfov_deg=hfov_deg,
        pose_kwargs=pose_kwargs,
        max_points=max_points
    )

    # --- local helpers (self-contained so only this function needs replacing) ---
    def _eval_spline_at_t(P_sample, t_sample, ts_query):
        """
        Fit/evaluate a cubic through (t_sample, P_sample) and eval at ts_query in [0,1].
        Uses SciPy CubicSpline if available; otherwise Catmull-Rom on the same t-domain.
        """
        P = np.asarray(P_sample, float)
        t = np.asarray(t_sample, float)
        tq = np.asarray(ts_query, float)

        if P.shape[0] == 0:
            return np.empty((0, 3), float)
        if P.shape[0] == 1:
            return np.repeat(P, repeats=len(tq), axis=0)

        # strictly increasing unique t
        t_u, idx = np.unique(t, return_index=True)
        P_u = P[idx]
        tq = np.clip(tq, 0.0, 1.0)

        try:
            from scipy.interpolate import CubicSpline  # local import
            if P_u.shape[0] >= 3:
                xs = CubicSpline(t_u, P_u[:, 0], bc_type="natural")(tq)
                ys = CubicSpline(t_u, P_u[:, 1], bc_type="natural")(tq)
                zs = CubicSpline(t_u, P_u[:, 2], bc_type="natural")(tq)
                return np.stack([xs, ys, zs], axis=1)
        except Exception:
            pass
        # fallback: Catmull-Rom on supplied t-knots
        return catmull_rom_sample(P_u, t_u, tq)

    def _fit_eval_segment(t_sample, P_sample, ts_dense):
        # sort by t, then evaluate
        t_sample = np.asarray(t_sample, float)
        P_sample = np.asarray(P_sample, float)
        order = np.argsort(t_sample)
        t_sorted = t_sample[order]
        P_sorted = P_sample[order]
        Y = _eval_spline_at_t(P_sorted, t_sorted, ts_dense)
        return dict(P=P_sorted, ts_dense=ts_dense, Y_dense=Y), Y

    # compute "orange" for a specific border pixel (i,j)
    def _orange_for_ij(i, j):
        p_ij = extras["xyz_img"][i, j]
        dir_vec, spline_len, _ = find_surface_point_unfold_params(
            np.asarray(extras["pcd"].points),          # surface_pts
            extras["surface_interior_cg"],             # SICG
            p_ij                                       # green on the surface
        )
        return unfold_surface_point(extras["surface_interior_cg"], dir_vec, spline_len)

    # --- 1) rebuild the filtered (i,j) list in the SAME order as the sampled items ---
    H, W = xyz_img.shape[:2]

    rc_sample = border_grid_indices((H, W))
    if max_points is not None and max_points > 0:
        rc_sample = rc_sample[:max_points]

    filtered_rc = []
    for (i, j) in rc_sample:
        p_ij = extras["xyz_img"][i, j]
        if not np.all(np.isfinite(p_ij)):
            continue
        if np.linalg.norm(p_ij - extras["surface_interior_cg"]) <= 1e-12:
            continue
        filtered_rc.append((i, j))

    # sanity: items were generated in this same filtered order
    assert len(filtered_rc) == len(extras["items"]), \
        "Mismatch between filtered border pixels and generated sample items."

    # collect the unfolded oranges in the same order (already produced)
    sample_unfolded_ordered = np.asarray([it["unfolded"] for it in extras["items"]], float)

    # --- 2) split samples by side, storing *t* per your traversal directions ---
    def side_of(i, j):
        if i == 0:     return "top"
        if i == H - 1: return "bottom"
        if j == 0:     return "left"
        if j == W - 1: return "right"
        return None

    side_groups = dict(top=[], bottom=[], left=[], right=[])
    for (i, j), P_orange in zip(filtered_rc, sample_unfolded_ordered):
        s = side_of(i, j)
        if s is None:
            continue
        # normalized coordinate along traversal for each side:
        if s == "top":      t =  j / (W - 1.0)                 # left -> right
        elif s == "right":  t =  i / (H - 1.0)                 # top -> bottom
        elif s == "bottom": t = (W - 1.0 - j) / (W - 1.0)      # right -> left
        else:               t = (H - 1.0 - i) / (H - 1.0)      # left: bottom -> top
        side_groups[s].append((t, P_orange))

    # --- 3) compute corner oranges (endpoints) ---
    tl = (0, 0)
    tr = (0, W - 1)
    br = (H - 1, W - 1)
    bl = (H - 1, 0)

    P_tl = _orange_for_ij(*tl)
    P_tr = _orange_for_ij(*tr)
    P_br = _orange_for_ij(*br)
    P_bl = _orange_for_ij(*bl)

    # --- 4) per-side corner→corner spline fit + dense evaluation ---
    splines = {}
    dense_red = {}

    # TOP: tl -> tr (t=j/(W-1))
    top_pts = sorted(side_groups["top"], key=lambda x: x[0])
    t_top = [0.0] + [t for (t, _) in top_pts] + [1.0]
    P_top = [P_tl] + [P for (_, P) in top_pts] + [P_tr]
    ts_top_dense = np.linspace(0.0, 1.0, W)
    splines["top"], dense_red["top"] = _fit_eval_segment(t_top, P_top, ts_top_dense)

    # RIGHT: tr -> br (t=i/(H-1))
    right_pts = sorted(side_groups["right"], key=lambda x: x[0])
    t_right = [0.0] + [t for (t, _) in right_pts] + [1.0]
    P_right = [P_tr] + [P for (_, P) in right_pts] + [P_br]
    ts_right_dense = np.linspace(0.0, 1.0, H)
    splines["right"], dense_red["right"] = _fit_eval_segment(t_right, P_right, ts_right_dense)

    # BOTTOM: br -> bl (t=(W-1-j)/(W-1))
    bottom_pts = sorted(side_groups["bottom"], key=lambda x: x[0])
    t_bottom = [0.0] + [t for (t, _) in bottom_pts] + [1.0]
    P_bottom = [P_br] + [P for (_, P) in bottom_pts] + [P_bl]
    ts_bottom_dense = np.linspace(0.0, 1.0, W)
    splines["bottom"], dense_red["bottom"] = _fit_eval_segment(t_bottom, P_bottom, ts_bottom_dense)

    # LEFT: bl -> tl (t=(H-1-i)/(H-1))
    left_pts = sorted(side_groups["left"], key=lambda x: x[0])
    t_left = [0.0] + [t for (t, _) in left_pts] + [1.0]
    P_left = [P_bl] + [P for (_, P) in left_pts] + [P_tl]
    ts_left_dense = np.linspace(0.0, 1.0, H)
    splines["left"], dense_red["left"] = _fit_eval_segment(t_left, P_left, ts_left_dense)

    # --- 5) pack results ---
    result = dict(
        unfolded_sample=np.asarray(sample_unfolded_ordered, dtype=np.float32),
        splines=splines,
        dense_red=dense_red,
    )

    return result, extras


# -------------------- NEW visualization that also draws tiny red dots --------------------
def visualize_scene_borders(pcd, cg, surface_interior_cg, items, red_points_by_side,
                            spline_color, show_green_to_orange=True):
    if not CONFIG.vis_debug:
        return

    draw = [pcd]
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_extent()) if bbox is not None else 1.0

    # Plane G + anchors
    plane_G = _make_plane_G(bbox, surface_interior_cg[2], color=[0.2, 0.6, 1.0])
    draw.append(plane_G)
    r_cg = max(1e-6, 0.0020 * diag)
    r_sicg = max(1e-6, 0.0018 * diag)
    draw.append(make_sphere(cg, r_cg, [1.0, 0.0, 0.0]))
    draw.append(make_sphere(surface_interior_cg, r_sicg, [0.0, 0.0, 1.0]))

    # Reuse the per-item rendering from visualize_scene_multi but inline it here
    r_point = max(1e-6, 0.0018 * diag)
    r_mid_purple = max(1e-6, 0.0015 * diag)
    r_mid_yellow = max(1e-6, 0.0015 * diag)
    r_unfolded = max(1e-6, 0.0018 * diag)
    r_radial_proj = max(1e-6, 0.0018 * diag)

    dash_len = 0.02 * diag
    gap_len  = 0.015 * diag

    for idx, it in enumerate(items, 1):
        point_i = it["point"]
        mids_purple = it.get("mids", np.empty((0,3)))
        mids_yellow = it.get("proj_points", np.empty((0,3)))
        curve = it.get("curve", None)
        unfolded_i = it.get("unfolded", None)
        radial_proj_i = it.get("radial_proj", None)

        draw.append(make_sphere(point_i, r_point, [0.0, 1.0, 0.0]))
        draw.append(make_lineset(np.vstack([surface_interior_cg, point_i]),
                                 np.array([[0, 1]], np.int32), [1.0, 1.0, 1.0]))
        for M in mids_purple:
            draw.append(make_sphere(M, r_mid_purple, [1.0, 0.0, 1.0]))
        for P in mids_yellow:
            draw.append(make_sphere(P, r_mid_yellow, [1.0, 1.0, 0.0]))

        if curve is not None and curve.shape[0] >= 2:
            ls = make_polyline(curve, CONFIG.spline_color)
            if ls is not None: draw.append(ls)
            p0, p1 = curve[0], curve[1]
            draw.append(make_arrow_from_to(p0, p1 + 19*(p1 - p0), [0.0, 1.0, 1.0], thin_factor=0.5))

        if unfolded_i is not None:
            draw.append(make_sphere(unfolded_i, r_unfolded, [1.0, 0.5, 0.0]))
            if show_green_to_orange:
                ds = _make_dashed_line(point_i, unfolded_i, [1.0, 0.5, 0.0], dash_len, gap_len)
                if ds is not None: draw.append(ds)

        if radial_proj_i is not None:
            draw.append(make_sphere(radial_proj_i, r_radial_proj, [0.6, 0.6, 0.6]))
            if show_green_to_orange:
                ds_g = _make_dashed_line(point_i, radial_proj_i, [0.6, 0.6, 0.6], dash_len, gap_len)
                if ds_g is not None: draw.append(ds_g)

    # Tiny red dots: as a point cloud to get the same tiny size as black pc
    all_red = []
    for s in ("top", "bottom", "left", "right"):
        Y = red_points_by_side.get(s, None)
        if Y is not None and len(Y) > 0:
            all_red.append(np.asarray(Y, dtype=float))
    if len(all_red) > 0:
        reds = np.vstack(all_red)
        red_pcd = o3d.geometry.PointCloud()
        red_pcd.points = o3d.utility.Vector3dVector(reds)
        red_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # tiny red points
        draw.append(red_pcd)

    o3d.visualization.draw_geometries(
        draw,
        window_name="Borders: orange splines + tiny red suggestions on plane G",
        width=1400, height=900
    )

def unfold_surface(xyz_img,
                   hfov_deg: float = 66.0,
                   pose_kwargs=None):
    """
    Unfold the entire surface using the fast row-wise interpolation method:

      1) Get dense-red border splines on plane G via unfold_image_borders(...)
      2) Orient sides to a consistent direction:
            top:    left->right
            bottom: left->right   (reverse input)
            left:   top->bottom   (reverse input)
            right:  top->bottom
      3) For each row I, compute a scalar E(I) in [0,1] as the average of:
            |L[I]-TL| / |TL-BL|  and  |R[I]-TR| / |TR-BR|
         (distances measured in XY; clamp to [0,1]).
      4) For each column J, set:
            S[I,J] = T[J] + E(I) * (B[J] - T[J])     (performed in 3D)

    Returns
    -------
    unfolded_surface : (H, W, 3) float32
        The unfolded surface points on plane G.
    context : dict
        {
          'pcd', 'cg', 'surface_interior_cg', 'xyz_img',
          'sides': {'top','right','bottom','left'}  # oriented 3D border samples
        }
    """
    import numpy as _np

    # 0) Get dense red borders + scene context (on plane G)
    borders, extras = unfold_image_borders(
        xyz_img=xyz_img,
        hfov_deg=hfov_deg,
        pose_kwargs=pose_kwargs,
        max_points=None,
    )
    dense_red = borders["dense_red"]  # dict: side -> (N,3)

    # ---- 1) Orient sides to consistent directions ----
    # Generator conventions:
    #   top:    left->right (already)
    #   right:  top->bottom (already)
    #   bottom: right->left  -> reverse to left->right
    #   left:   bottom->top  -> reverse to top->bottom
    top    = _np.asarray(dense_red["top"],    dtype=_np.float32)            # (W,3)
    right  = _np.asarray(dense_red["right"],  dtype=_np.float32)            # (H,3)
    bottom = _np.asarray(dense_red["bottom"], dtype=_np.float32)[::-1].copy()
    left   = _np.asarray(dense_red["left"],   dtype=_np.float32)[::-1].copy()

    W = int(top.shape[0])
    H = int(right.shape[0])
    if bottom.shape[0] != W or left.shape[0] != H:
        raise ValueError("dense_red border lengths are inconsistent with image size.")

    # ---- 2) Compute per-row scalar E(I) from XY distances (clamped to [0,1]) ----
    # Corners in XY
    top_xy    = top[:, :2]
    bottom_xy = bottom[:, :2]
    left_xy   = left[:, :2]
    right_xy  = right[:, :2]

    TL, TR = top_xy[0],    top_xy[-1]
    BL, BR = bottom_xy[0], bottom_xy[-1]

    TL_BL_len = float(_np.linalg.norm(TL - BL))
    TR_BR_len = float(_np.linalg.norm(TR - BR))
    if TL_BL_len <= 0.0: TL_BL_len = 1.0
    if TR_BR_len <= 0.0: TR_BR_len = 1.0

    E1 = _np.linalg.norm(left_xy  - TL, axis=1) / TL_BL_len   # (H,)
    E2 = _np.linalg.norm(right_xy - TR, axis=1) / TR_BR_len   # (H,)
    E  = _np.clip(0.5 * (E1 + E2), 0.0, 1.0).astype(_np.float32)  # (H,)

    # ---- 3) Column-wise 3D interpolation from T(J) to B(J) using E(I) ----
    T3 = top[None, :, :]                 # (1, W, 3)
    dTB = (bottom - top)[None, :, :]     # (1, W, 3)
    Ecol = E[:, None, None]              # (H, 1, 1)

    S = T3 + Ecol * dTB                  # (H, W, 3) — on plane G
    unfolded_surface = S.astype(_np.float32, copy=False)

    # ---- 4) Pack context (useful for debug/overlays) ----
    sides = dict(top=top, right=right, bottom=bottom, left=left)
    context = dict(
        pcd=extras["pcd"],
        surface_interior_cg=extras["surface_interior_cg"],
        cg=extras["cg"],
        xyz_img=extras["xyz_img"],
        sides=sides,
    )
    return unfolded_surface, context

def move_depth(depth_image, bg_image, gep_image):
    assert depth_image.dtype == np.float32 and bg_image.dtype == np.float32 and \
        gep_image.dtype == np.float32
    relative_depth_before = depth_image / bg_image

    # Convert depth data into proximity data
    proximity_image = 255.0 - depth_image
    bg_prox = 255.0 - bg_image
    gep_prox = 255.0 - gep_image

    # Extract foreground
    fg_prox = proximity_image - bg_prox
    assert np.all(fg_prox >= 0)

    # Rescale foreground objects
    # 1. No rescaling
    # r = 1
    # 2. Rescaling based on relative depth
    # r = gep_image[min_depth_index] * (1 - relative_depth_before[min_depth_index]) / fg_prox[min_depth_index]
    # 3. Rescaling based on relative proximity
    # r = (gep_prox[min_depth_index] * (1 - relative_prox_before[min_depth_index])) / \
    #       (fg_prox[min_depth_index] * relative_prox_before[min_depth_index])
    # fg_prox_scaled = fg_prox * r
    r = np.where(fg_prox != 0, gep_image * (1 - relative_depth_before) / fg_prox, 0.0)
    fg_prox_scaled = fg_prox * r

    # Replace old background with new background
    # 1. main functionality
    moved_prox = fg_prox_scaled + gep_prox
    # 2. No background
    # moved_prox = fg_prox_scaled
    # 3. No rescaling
    # moved_prox = fg_prox_scaled + bg_prox

    # Check
    # relative_prox_before = bg_prox / proximity_image
    # relative_prox_after = gep_prox / moved_prox
    # relative_depth_before = depth_image / gep_image
    # relative_depth_after = moved_depth / gep_image
    # print(relative_depth_before[min_depth_index], relative_depth_after[min_depth_index])
    # print(relative_prox_before[min_depth_index], relative_prox_after[min_depth_index])

    assert np.max(moved_prox) < 255.0 and np.min(moved_prox) >= 0.0,\
        f"min: {np.min(moved_prox):.2f}, max: {np.max(moved_prox):.2f}"
    moved_depth = 255.0 - moved_prox
    return moved_depth

def calc_ground_depth(hfov_degs,
                      output_shape,
                      p,
                      fixed_alt=10.0,
                      horizon_pitch_rad=-0.034):
    """
    Returns a float32 image whose pixel intensities encode
        C · metric_distance_to_ground    0 ≤ value ≤ 255

    C is chosen so that the farthest *ground* distance visible in
    the image is mapped to exactly 255.  Rays that never intersect
    the ground, or whose pitch > horizon_pitch_rad, are forced to 255.
    """
    H, W = output_shape

    # --- 1) intrinsics ------------------------------------------------
    f  = (W / 2) / np.tan(np.radians(hfov_degs) / 2)
    cx, cy = W / 2.0, H / 2.0

    # --- 2) unit rays in camera coords --------------------------------
    xg, yg = np.meshgrid(np.arange(W), np.arange(H))
    X = (xg - cx) / f
    Y = (yg - cy) / f
    Z = np.ones_like(X)
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    # dirs_cam = np.stack([X / norm, Y / norm, Z / norm], axis=-1)
    dirs = np.stack([X / norm, Y / norm, Z / norm], axis=-1)
    dirs_nwu = dirs @ p.getCAM2NWU().T

    # # --- 3) camera  ->  NED -------------------------------------------
    # Ry90neg = np.array([[ 0, 0, 1],
    #                     [ 0, 1, 0],
    #                     [-1, 0, 0]])
    # Rx90neg = np.array([[1, 0,  0],
    #                     [0, 0, -1],
    #                     [0, 1,  0]])
    # dirs_ned = dirs_cam @ (Rx90neg @ Ry90neg).T

    # # --- 4) apply camera pitch about +Y in NED ------------------------
    # pitch_rad = -1.57
    # Ry = np.array([[ np.cos(-pitch_rad), 0, -np.sin(-pitch_rad)],
    #                [ 0,                  1,  0               ],
    #                [ np.sin(-pitch_rad), 0,  np.cos(-pitch_rad)]])
    # dirs_ = dirs_ned @ Ry.T                     # final unit directions
    # dirs = dirs_cam @ (rotation_matrix_x(np.pi).T @ p.getCAM2NWU().T)

    # --- 5) ray pitch angle & ground intersection --------------------
    horiz_len     = np.linalg.norm(dirs[..., :2], axis=-1)
    pitch_angle   = np.arctan2(-dirs[..., 2], horiz_len)   # +ve up, –ve down
    dir_z         = dirs[..., 2]                           # downward component

    eps           = 1e-12
    distance_raw  = np.where(dir_z > eps,                 # t = alt / dir_z
                             fixed_alt / dir_z,
                             np.inf).astype(np.float32)

    # --- 6) classify pixels ------------------------------------------
    sky_mask      = (pitch_angle > horizon_pitch_rad) | ~np.isfinite(distance_raw)
    ground_mask   = ~sky_mask                            # valid finite hits

    if not np.any(ground_mask):
        raise RuntimeError("Camera sees no ground below the horizon!")

    d_max         = distance_raw[ground_mask].max()      # farthest ground distance
    scale         = 255.0 / d_max                        # C  so that d_max→255

    # --- 7) build output image ---------------------------------------
    depth_img     = np.full((H, W), 255.0, dtype=np.float32)    # start with sky
    depth_img[ground_mask] = distance_raw[ground_mask] * scale
    depth_img     = np.minimum(depth_img, 255.0)                # clamp just in case
    return depth_img

def NDFDrop_depth(dpc, bpcd, gpc):
    dpc_pcd = pcm2pcd(dpc)
    # bpc_pcd = pcm2pcd(bpc)
    bpc_pcd = bpcd
    gpc_pcd = pcm2pcd(gpc)
    
    bpc_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    bpc_pcd = bpc_pcd.voxel_down_sample(0.02)
    gpc_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    gpc_pcd = gpc_pcd.voxel_down_sample(0.02)
    
    proj_b = project_points_multi_fast(bpc_pcd, dpc, k=8, just_proj=True)
    proj_g = project_points_multi_fast(gpc_pcd, proj_b, k=8, just_proj=True)
    dist = euclidean_distance_map(dpc, proj_b)
    unfolded_dpc = proj_g.copy()
    unfolded_dpc[:, :, 2:3] += dist
    return unfolded_dpc, proj_g[:,:,2]

def calc_dist2bottom(pc_up: np.ndarray, pc_down: np.ndarray) -> np.ndarray:
    H, W, _ = pc_up.shape
    up_flat = pc_up.reshape(-1, 3)
    down_flat = pc_down.reshape(-1, 3)

    xy_down = down_flat[:, :2]
    z_down = down_flat[:, 2]
    interp = LinearNDInterpolator(xy_down, z_down, fill_value=np.nan)

    projected_z = np.empty_like(up_flat[:, 2])
    for i, (x, y, _) in enumerate(up_flat):
        z_proj = interp(x, y)
        if np.isnan(z_proj):
            dists = np.linalg.norm(xy_down - [x, y], axis=1)
            z_proj = z_down[np.argmin(dists)]
        projected_z[i] = z_proj

    vertical_distances = up_flat[:, 2] - projected_z
    return vertical_distances.reshape((H, W))

def drop_depth(dpc, bpc, gpc):
    alt_diff_db = calc_dist2bottom(dpc, bpc)
    mean_gpc_z = np.mean(gpc[:,:,2])
    alt_diff_dg = dpc[:,:,2] - mean_gpc_z
    diff = -(alt_diff_db - alt_diff_dg)
    dropped_dpc = dpc.copy()
    dropped_dpc[:,:,2] -= diff
    return dropped_dpc

def rescale_depth(depth_image, bg_image, pitch):
    gep_f = calc_ground_depth(66.0, Pose, output_shape=depth_image.shape)
    # Convert to float32
    depth_f = depth_image.astype(np.float32)
    bg_f = bg_image.astype(np.float32)
    # gep_f = gep_depth.astype(np.float32)
    # Compute how much closer foreground is, in original image
    fg_raw_f = depth_f - bg_f
    np.clip(fg_raw_f, 0, 255, out=fg_raw_f)
    dist_to_camera_raw = np.abs(255.0 - bg_f)
    # Avoid divide-by-zero
    dist_to_camera_raw[dist_to_camera_raw == 0] = 1.0
    # Ratio of foreground proximity
    ratio = fg_raw_f / dist_to_camera_raw
    if np.min(ratio) < 0 or np.max(ratio) > 1:
        raise ValueError("Non-logical ratio calculation")

    dist_to_camera_new = np.abs(255.0 - gep_f)
    dist_to_camera_new[dist_to_camera_new == 0] = 1.0
    # Apply same ratio to refined background to get final proximity
    fg_new_f = ratio * dist_to_camera_new
    final_f = gep_f + fg_new_f
    if np.min(final_f) < 0 or np.max(final_f) > 255.0:
        raise ValueError("Non-logical ratio calculation")
    return final_f, gep_f

def unfold_depth(dpc, bpc, gpc):
    nan_indices = np.argwhere(np.isnan(bpc).any(axis=2))
    dpc_pcd = pcm2pcd(dpc)
    bpc_pcd = pcm2pcd(bpc)
    gpc_pcd = pcm2pcd(gpc)
    if len(bpc_pcd.points) == 0:
        raise ValueError("bpc_pcd is empty, cannot proceed.")
    bpc_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    bpc_pcd = bpc_pcd.voxel_down_sample(0.02)
    gpc_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    gpc_pcd = gpc_pcd.voxel_down_sample(0.02)
    proj_b = project_points_multi_fast(bpc_pcd, dpc, k=8, just_proj=True)
    proj_g, _ = unfold_surface(bpc, hfov_deg=80.0)
    proj_b_idx = map_pc_proj_to_index(proj_b, bpc) # H * W * 2
    proj_b_idx_i = np.int32(proj_b_idx)
    dist = euclidean_distance_map(dpc, proj_b)     # H * W * 1
    unfolded_dpc = np.zeros_like(proj_g)
    for i in range(dpc.shape[0]):
        for j in range(dpc.shape[1]):
            unfolded_dpc[i, j, 0] = proj_g[proj_b_idx_i[i,j,0], proj_b_idx_i[i,j,1], 0]
            unfolded_dpc[i, j, 1] = proj_g[proj_b_idx_i[i,j,0], proj_b_idx_i[i,j,1], 1]
    nan_indices = np.argwhere(np.isnan(unfolded_dpc).any(axis=2))
    return unfolded_dpc

def map_pc_proj_to_index(pc_proj: np.ndarray, pc: np.ndarray, k: int = 3) -> np.ndarray:
    nan_val = 1e9  # large number to replace invalid points
    
    H_pc, W_pc, _ = pc.shape
    H_proj, W_proj, _ = pc_proj.shape

    # Replace NaNs/Infs in pc with large number
    pc_safe = np.where(np.isfinite(pc), pc, nan_val)

    # Flatten pc
    pc_flat = pc_safe.reshape(-1, 3)
    
    # Create grid coordinates
    h_coords, w_coords = np.meshgrid(np.arange(H_pc), np.arange(W_pc), indexing='ij')
    coords_flat = np.stack([h_coords, w_coords], axis=-1).reshape(-1, 2)

    # Build KDTree on all points
    tree = cKDTree(pc_flat)

    # Flatten projected points
    proj_flat = pc_proj.reshape(-1, 3)

    # Handle invalid query points
    valid_mask = np.isfinite(proj_flat).all(axis=1)
    mapped_coords_flat = np.zeros((proj_flat.shape[0], 2), dtype=float)

    if np.any(valid_mask):
        # Query only valid points
        dists, idxs = tree.query(proj_flat[valid_mask], k=k)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # Inverse distance weighting
        dists = np.maximum(dists, 1e-8)
        weights = 1.0 / dists
        weights /= np.sum(weights, axis=1, keepdims=True)

        # Weighted average of indices
        mapped_coords_flat[valid_mask] = np.sum(coords_flat[idxs] * weights[..., None], axis=1)

    # Clip to valid range
    mapped_coords_flat[:, 0] = np.clip(mapped_coords_flat[:, 0], 0, H_pc-1)
    mapped_coords_flat[:, 1] = np.clip(mapped_coords_flat[:, 1], 0, W_pc-1)

    # Reshape back to H_proj, W_proj
    return mapped_coords_flat.reshape(H_proj, W_proj, 2)

def depyramidize_pointCloud(points: np.ndarray, eps: float = 1e-12):
    """
    Vectorized depyramidization for a whole point cloud.

    For each point P:
      - Intersect L(t) = t * P with plane z = plane_z → I (blue)
        * If |Pz| > eps: t = plane_z / Pz ; I = t * P
        * Else:
            - If |plane_z| <= eps: I = (0,0,0)
            - Else: I = (Px, Py, plane_z)     # lateral fallback
      - point2plane  = ||P - I||
      - plane2origin = ||I||                  # <— CHANGED from ||P||
      - dep_point  = (Ix, Iy, plane_z + point2plane)
      - dep_origin = (Ix, Iy, plane_z + plane2origin)  # <— CHANGED

    Returns:
      dep_points:  (N,3)
      dep_origins: (N,3)
      info: dict with:
        - blue: (N,3) intersections I
        - t: (N,) line parameters (NaN for fallback)
        - valid_mask: (N,) bool where |Pz|>eps
        - point2plane:  (N,)
        - plane2origin: (N,)   # <— ADDED
    """
    P = np.asarray(points, dtype=np.float64)
    plane_z = float(np.min(P[:, 2]))
    assert P.ndim == 2 and P.shape[1] == 3, "points must be (N,3)"

    N = P.shape[0]
    Pz = P[:, 2]
    valid_mask = np.abs(Pz) > eps

    t = np.full((N,), np.nan, dtype=np.float64)
    t[valid_mask] = plane_z / Pz[valid_mask]

    I = np.empty_like(P)
    I[valid_mask] = (t[valid_mask, None] * P[valid_mask])

    inv_mask = ~valid_mask
    if np.any(inv_mask):
        I[inv_mask, :2] = P[inv_mask, :2]
        if abs(plane_z) <= eps:
            I[inv_mask] = 0.0  # origin
        else:
            I[inv_mask, 2] = plane_z

    point2plane  = np.linalg.norm(P - I, axis=1)
    plane2origin = np.linalg.norm(I, axis=1)  # <— CHANGED

    dep_points = np.empty_like(P)
    dep_origins = np.empty_like(P)

    dep_points[:, 0:2] = I[:, 0:2]
    dep_points[:, 2]   = plane_z + point2plane

    dep_origins[:, 0:2] = I[:, 0:2]
    dep_origins[:, 2]   = plane_z + plane2origin  # <— CHANGED

    info = dict(
        blue=I,
        t=t,
        valid_mask=valid_mask,
        point2plane=point2plane,
        plane2origin=plane2origin,   # <— ADDED
    )
    return dep_points, dep_origins, info

def downsample_pcm(pcm: np.ndarray, W_final: int) -> np.ndarray:
    """
    Downsample a PCM (H, W, 3) numpy array to a target width, keeping aspect ratio.
    
    Args:
        pcm (np.ndarray): Input PCM array of shape (H, W, 3)
        W_final (int): Desired final width after downsampling
        
    Returns:
        np.ndarray: Downsampled PCM array (H_new, W_final, 3)
    """
    # --- Validation ---
    if not isinstance(pcm, np.ndarray):
        raise TypeError("pcm must be a numpy array")
    if pcm.ndim != 3 or pcm.shape[2] != 3:
        raise ValueError("pcm must have shape (H, W, 3)")
    if W_final <= 0:
        raise ValueError("W_final must be a positive integer")

    H, W, _ = pcm.shape

    # --- Compute new size while keeping aspect ratio ---
    ratio = W_final / W
    H_final = int(H * ratio)

    # --- Downsample using OpenCV with area interpolation ---
    pcm_down = cv2.resize(pcm, (W_final, H_final), interpolation=cv2.INTER_AREA)

    return pcm_down
