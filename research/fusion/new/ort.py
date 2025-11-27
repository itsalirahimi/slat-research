import cv2
import numpy as np
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")

from utils.conversion import pcd2pcm
from ioHandle.IOHandler import load_pcd, save_pcm_as_pcd

def _finite_xy_from_grid(grid3d: np.ndarray) -> np.ndarray:
    """
    Flatten (H, W, 3) -> (N, 2) and drop non-finite xy.
    """
    pts = grid3d.reshape(-1, 3)
    xy = pts[:, :2]
    mask = np.isfinite(xy).all(axis=1)
    xy = xy[mask]
    if xy.shape[0] < 4:
        raise ValueError("Not enough finite points to define a bounding box (need >= 4).")
    return xy


def _bbox_corners(xy: np.ndarray) -> np.ndarray:
    """
    Given (N, 2) xy points, return 4 corners of the axis-aligned bounding box:
    order: [minx,miny], [maxx,miny], [maxx,maxy], [minx,maxy]
    """
    minx, miny = xy.min(axis=0)
    maxx, maxy = xy.max(axis=0)
    return np.array([
        [minx, miny],  # bottom-left
        [maxx, miny],  # bottom-right
        [maxx, maxy],  # top-right
        [minx, maxy],  # top-left
    ], dtype=float)


def _estimate_rigid_2d(src: np.ndarray, dst: np.ndarray):
    """
    Estimate 2D rigid transform (rotation R, translation t) such that:

        dst ≈ R @ src + t

    src, dst: (N, 2)
    Returns:
        R: (2, 2) rotation matrix
        t: (2,) translation vector
    """
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src and dst must both be (N, 2).")

    # centroids
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    X0 = src - src_mean
    Y0 = dst - dst_mean

    # cross-covariance
    H = X0.T @ Y0  # (2,2)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T  # rotation

    # enforce proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # translation: dst_mean = R @ src_mean + t
    t = dst_mean - R @ src_mean

    return R, t


def align_surface_gep_gt(
    gep: np.ndarray,
    gt: np.ndarray,
    use_rotation: bool = True,
):
    """
    Align gt surface to gep surface using translation in 3D and optional
    rotation in XY (no scaling).

    - If use_rotation=True:
        gt_xy' = R @ gt_xy + t_xy
      where R is 2x2 rotation, t_xy is 2D translation.
    - If use_rotation=False:
        R = I, and t_xy is chosen to best match the 4 corners (centroid align).
    - Then compute tz to align mean Z of gt to mean Z of gep.
    - Apply XYZ translation to all points.

    Parameters
    ----------
    gep : (H, W, 3) ndarray
        Reference surface (e.g. "gep").
    gt : (H, W, 3) ndarray
        Surface to be aligned onto gep (e.g. "gt").
    use_rotation : bool, optional
        If True, estimate XY rotation + translation from corners.
        If False, only translation (no rotation). Default True.

    Returns
    -------
    gt_aligned : (H, W, 3) ndarray
        Transformed gt (XY rotated+translated if use_rotation, Z translated).
    translation_xyz : (3,) ndarray
        3D translation vector [tx, ty, tz].
    rotation_rad : float
        Rotation angle in radians (CCW) applied in XY plane.
        0.0 if use_rotation=False.
    R : (2, 2) ndarray
        2D rotation matrix used in XY (identity if use_rotation=False).
    corners_gep : (4, 2) ndarray
        XY corners of gep bounding box.
    corners_gt : (4, 2) ndarray
        XY corners of gt bounding box (before transform).
    """
    if gep.shape != gt.shape or gep.shape[-1] != 3:
        raise ValueError("gep and gt must both be (H, W, 3) arrays with same shape.")

    # ---------- 1) Corners in XY ----------
    gep_xy = _finite_xy_from_grid(gep)
    gt_xy = _finite_xy_from_grid(gt)

    corners_gep = _bbox_corners(gep_xy)  # (4, 2)
    corners_gt = _bbox_corners(gt_xy)    # (4, 2)

    # ---------- 2) Rotation + translation in XY (no scale) ----------
    if use_rotation:
        R, t_xy = _estimate_rigid_2d(corners_gt, corners_gep)
    else:
        R = np.eye(2, dtype=float)
        # best pure translation: align centroids of corners
        t_xy = (corners_gep - corners_gt).mean(axis=0)

    H, W, _ = gt.shape
    gep_flat = gep.reshape(-1, 3)
    gt_flat = gt.reshape(-1, 3).copy()

    # apply XY transform
    xy = gt_flat[:, :2]
    xy_rot = (R @ xy.T).T      # (N, 2)
    xy_trans = xy_rot + t_xy   # (N, 2)
    gt_flat[:, :2] = xy_trans

    # ---------- 3) Z translation ----------
    # Align mean Z of gt to mean Z of gep (using finite values only)
    mask_z = np.isfinite(gep_flat[:, 2]) & np.isfinite(gt_flat[:, 2])
    if np.any(mask_z):
        gep_z_mean = gep_flat[mask_z, 2].mean()
        gt_z_mean = gt_flat[mask_z, 2].mean()
        tz = float(gep_z_mean - gt_z_mean)
    else:
        tz = 0.0

    gt_flat[:, 2] += tz
    gt_aligned = gt_flat.reshape(H, W, 3)

    # rotation angle (for convenience)
    rotation_rad = float(np.arctan2(R[1, 0], R[0, 0]))

    translation_xyz = np.array([t_xy[0], t_xy[1], tz], dtype=float)

    return gt_aligned, translation_xyz, rotation_rad, R, corners_gep, corners_gt

model = load_pcd("data/ortholoc/fusion/de/fused/L08_R0000.pcd")
H, W = 273, 365
mpcm = pcd2pcm(model, H, W)
H, W = 767, 1024
cimg_ds = cv2.resize(mpcm, (W, H), interpolation=cv2.INTER_AREA)

# gt = np.load("/home/psash/repos/OrthoLoc/sample/L08_R0000/point_map.npy").astype(np.float32)

gtc = pcd2pcm(load_pcd("data/ortholoc/data/gt_cam/L08_R0000.pcd"), H, W)
# # align with rotation
# gt_aligned, t_xyz_rot, rot_rad, R, c_gep, c_gt = align_surface_gep_gt(cimg_ds, gt, use_rotation=True)

# align with pure translation (no rotation)
gt_aligned, t_xyz_tr, rot_rad_tr, R_tr, _, _ = align_surface_gep_gt(cimg_ds, gtc, use_rotation=False)

print(gt_aligned.shape)
save_pcm_as_pcd(gt_aligned, "gt_aligned.pcd", color = [255, 0, 0])
save_pcm_as_pcd(gtc, "gt_org.pcd", color = [0, 255, 0])

