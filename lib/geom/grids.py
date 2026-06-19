import numpy as np

# infer grid sizes (unique Xs and Ys with tolerance)
def unique_sorted_with_tol(a, atol):
    a_sorted = np.sort(a)
    uniq = [a_sorted[0]]
    for v in a_sorted[1:]:
        if abs(v - uniq[-1]) > atol:
            uniq.append(v)
    return np.asarray(uniq, dtype=float)

def infer_grid(ctrl_points,
        tol: float = 0.59):
    P = np.asarray(ctrl_points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.size == 0:
        raise ValueError("ctrl_points must be a non-empty (N,3) array")
    xs, ys = P[:, 0], P[:, 1]

    # extents
    W = float(xs.max() - xs.min())
    H = float(ys.max() - ys.min())

    tol_x = max(tol, 1e-8 * max(1.0, W))
    tol_y = max(tol, 1e-8 * max(1.0, H))
    ux = unique_sorted_with_tol(xs, tol_x)  # gw
    uy = unique_sorted_with_tol(ys, tol_y)  # gh
    gw, gh = len(ux), len(uy)
    if gw * gh != len(P):
        raise ValueError(f"Control net is not a full grid: {len(P)} pts vs {gw}×{gh}")
    return gw, gh

def reorder_ctrl_points_rowmajor(ctrl_pts: np.ndarray) -> np.ndarray:
    """Reorder control points to row-major by Y (rows) then X (cols)."""
    xs, ys = ctrl_pts[:, 0], ctrl_pts[:, 1]
    order = np.lexsort((xs, ys))  # primary: y, secondary: x
    return ctrl_pts[order]

def extract_pcm_corners(points_3d: np.ndarray) -> np.ndarray:
    """HxWx3 -> 4 corners in order: TL, TR, BR, BL"""
    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError("Input must be an HxWx3 numpy array")
    H, W, _ = points_3d.shape
    return np.array([
        points_3d[0, 0],          # TL
        points_3d[0, W - 1],      # TR
        points_3d[H - 1, W - 1],  # BR
        points_3d[H - 1, 0],      # BL
    ], dtype=float)