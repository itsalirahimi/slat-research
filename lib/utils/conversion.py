import numpy as np, open3d as o3d
from typing import Optional, Union

ArrayLike = Union[np.ndarray, list, tuple]


def pcd2pcdArr(pcd):
    return np.asarray(pcd.points, dtype=float)

def pcm2pcdArr(pcm):
    return pcm.reshape(-1, 3)

def pcm2pcd(pcm, visualization_image=None):
    points = pcm2pcdArr(pcm)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if (visualization_image is not None):
        h, w, _ = visualization_image.shape
        assert pcm.shape[:2] == (h, w), "Image and point cloud dimensions mismatch."
        colors = visualization_image.reshape(-1, 3).astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, [2,1,0]])
    return pcd

def pcdArr2pcd(points_xyz: ArrayLike, colors: Optional[ArrayLike] = None) -> o3d.geometry.PointCloud:
    """
    Build an Open3D PointCloud from arrays.

    Parameters
    ----------
    points_xyz : (N,3) array-like of float
        Point positions.
    colors : None | (3,) | (N,3) array-like of float [0..1] or uint8 [0..255]
        If (3,), a single RGB color is broadcast to all points.

    Returns
    -------
    o3d.geometry.PointCloud
    """
    # points
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"`points_xyz` must be shape (N,3); got {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError("`points_xyz` contains NaN/Inf.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)

    # colors (optional)
    if colors is not None:
        C = np.asarray(colors)
        if C.ndim == 1 and C.shape[0] == 3:
            # single color -> broadcast
            C = np.broadcast_to(C.reshape(1, 3), (P.shape[0], 3))
        elif C.ndim == 2 and C.shape == (P.shape[0], 3):
            pass
        else:
            raise ValueError(
                f"`colors` must be (3,) or (N,3) to match points; got {C.shape}"
            )

        # convert to float [0,1]
        if np.issubdtype(C.dtype, np.integer) or C.max() > 1.0:
            C = C.astype(float) / 255.0
        else:
            C = C.astype(float)

        # clamp just in case
        C = np.clip(C, 0.0, 1.0)

        if not np.isfinite(C).all():
            raise ValueError("`colors` contains NaN/Inf.")

        pcd.colors = o3d.utility.Vector3dVector(C)

    return pcd

def pcd2pcm(pcd: o3d.geometry.PointCloud, H: int, W: int) -> np.ndarray:
    """
    Convert a point cloud to an H x W x 3 numpy array.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        H (int): Height of the output grid.
        W (int): Width of the output grid.

    Returns:
        pcm (np.ndarray): Array of shape (H, W, 3) with color values in [0,1].
    """
    return pcdArr2pcm(pts = np.asarray(pcd.points, dtype=np.float32), H=H, W=W)

def pcdArr2pcm(pts: ArrayLike, H: int, W: int) -> np.ndarray:
    """
    Convert a point cloud to an H x W x 3 numpy array.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        H (int): Height of the output grid.
        W (int): Width of the output grid.

    Returns:
        pcm (np.ndarray): Array of shape (H, W, 3) with color values in [0,1].
    """
    if pts.shape[0] != H * W:
        raise ValueError(f"Point cloud has {pts.shape[0]} points, "
                         f"but expected {H*W} for reshaping.")

    # Reshape into image-like grid
    pcm = pts.reshape((H, W, 3))
    return pcm

def pcd2hw1(pcd: o3d.geometry.PointCloud, H: int, W: int) -> np.ndarray:
    _pcm = pcd2pcm(pcd, H, W)
    _norm = np.linalg.norm(_pcm, axis=2)    # (H, W)
    _norm = _norm[..., np.newaxis]          # (H, W, 1)
    return _norm