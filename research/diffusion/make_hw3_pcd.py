import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def _estimate_axis(values, target_len):
    """
    Estimate full regular axis coordinates from partial coordinate values.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    unique_vals = np.unique(np.round(values, 8))
    unique_vals.sort()

    if len(unique_vals) == target_len:
        return unique_vals

    if len(unique_vals) < 2:
        raise ValueError("Cannot estimate grid axis from fewer than 2 unique values.")

    diffs = np.diff(unique_vals)
    diffs = diffs[diffs > 1e-12]

    if len(diffs) == 0:
        raise ValueError("Cannot estimate grid spacing.")

    step = float(np.median(diffs))

    # Assumption: observed min belongs to the first column/row.
    # Without filename extent, missing full border rows/cols cannot be known exactly.
    start = float(unique_vals[0])

    return start + step * np.arange(target_len, dtype=np.float64)


def _fill_missing_z_idw(z_grid, k=8, power=2.0):
    """
    Fill NaN z cells using IDW interpolation in grid row/col space.
    """
    z_grid = z_grid.copy()

    missing = ~np.isfinite(z_grid)

    if not missing.any():
        return z_grid

    valid = ~missing

    if not valid.any():
        raise ValueError("All z values are missing; cannot interpolate.")

    valid_rc = np.column_stack(np.nonzero(valid))
    missing_rc = np.column_stack(np.nonzero(missing))

    valid_z = z_grid[valid]

    tree = cKDTree(valid_rc)

    kk = min(k, len(valid_z))

    try:
        dist, idx = tree.query(missing_rc, k=kk, workers=-1)
    except TypeError:
        dist, idx = tree.query(missing_rc, k=kk)

    if kk == 1:
        filled = valid_z[idx]
    else:
        dist = np.maximum(dist, 1e-12)
        weights = 1.0 / (dist ** power)
        filled = np.sum(weights * valid_z[idx], axis=1) / np.sum(weights, axis=1)

    rows = missing_rc[:, 0]
    cols = missing_rc[:, 1]

    z_grid[rows, cols] = filled

    return z_grid


def make_full_pcd(
    o3d_pcd_obj,
    H,
    W,
    origin="upper",
    duplicate_strategy="max",
    interpolation_k=8,
):
    """
    Convert an incomplete Open3D PCD object into a full H*W point cloud.

    Parameters
    ----------
    o3d_pcd_obj:
        open3d.geometry.PointCloud object with x/y/z points.

    H, W:
        Target dense grid size.
        For square RGB tiles, use H=A, W=A.

    origin:
        "upper" means row 0 is max Y, normal image/raster convention.
        "lower" means row 0 is min Y.

    duplicate_strategy:
        "max"  -> keep highest z when multiple points map to same pixel.
                  Good for voxel-top / DSM data.
        "mean" -> average duplicate z values.

    Returns
    -------
    full_pcd:
        New open3d.geometry.PointCloud with exactly H*W points.
    """
    points = np.asarray(o3d_pcd_obj.points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input point cloud must contain Nx3 xyz points.")

    points = points[np.isfinite(points).all(axis=1)]

    if len(points) == 0:
        raise ValueError("Input point cloud has no valid points.")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Estimate dense x/y axes from the partial point cloud
    x_axis = _estimate_axis(x, W)
    y_axis_inc = _estimate_axis(y, H)

    if origin == "upper":
        y_axis = y_axis_inc[::-1]
    elif origin == "lower":
        y_axis = y_axis_inc
    else:
        raise ValueError("origin must be 'upper' or 'lower'.")

    dx = float(np.median(np.diff(x_axis)))
    dy = float(abs(np.median(np.diff(y_axis))))

    x0 = x_axis[0]
    y0 = y_axis[0]

    cols_f = (x - x0) / dx

    if origin == "upper":
        rows_f = (y0 - y) / dy
    else:
        rows_f = (y - y0) / dy

    cols = np.rint(cols_f).astype(np.int64)
    rows = np.rint(rows_f).astype(np.int64)

    inside = (
        (rows >= 0) & (rows < H) &
        (cols >= 0) & (cols < W) &
        np.isfinite(z)
    )

    rows = rows[inside]
    cols = cols[inside]
    z = z[inside]

    if len(z) == 0:
        raise ValueError("No input points mapped inside the target HxW grid.")

    flat = rows * W + cols
    total = H * W

    if duplicate_strategy == "max":
        z_flat = np.full(total, -np.inf, dtype=np.float64)
        np.maximum.at(z_flat, flat, z)
        z_flat[z_flat == -np.inf] = np.nan

    elif duplicate_strategy == "mean":
        sums = np.bincount(flat, weights=z, minlength=total)
        counts = np.bincount(flat, minlength=total)

        z_flat = np.full(total, np.nan, dtype=np.float64)
        ok = counts > 0
        z_flat[ok] = sums[ok] / counts[ok]

    else:
        raise ValueError("duplicate_strategy must be 'max' or 'mean'.")

    z_grid = z_flat.reshape(H, W)

    z_grid = _fill_missing_z_idw(
        z_grid,
        k=interpolation_k,
        power=2.0,
    )

    xx, yy = np.meshgrid(x_axis, y_axis)

    full_points = np.column_stack([
        xx.reshape(-1),
        yy.reshape(-1),
        z_grid.reshape(-1),
    ])

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(full_points)

    return full_pcd


pcd = o3d.io.read_point_cloud("data/dsm2dtm/projection/dsm/canonical/tile_ix000_iy000_E152000p00-152500p00_N462000p00-462500p00.pcd")
img = cv2.imread("data/dsm2dtm/projection/dsm/rgb/tile_ix000_iy000_E152000p00-152500p00_N462000p00-462500p00.png")

H, W, _ = img.shape
full_pcd = make_full_pcd(
    pcd,
    H=H,
    W=W,
    origin="upper",
    duplicate_strategy="max",
    interpolation_k=8,
)

o3d.io.write_point_cloud("mini.pcd", pcd, write_ascii=False)
o3d.io.write_point_cloud("input_full.pcd", full_pcd, write_ascii=False)

print(np.asarray(full_pcd.points).shape)
# (A*A, 3)