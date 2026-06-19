import numpy as np

def rotation_matrix_x(phi):
    """Rotation about x-axis."""
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def rotation_matrix_y(theta):
    """Rotation about y-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def rotation_matrix_z(psi):
    """Rotation about z-axis."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

def transform_kps_nwu2camera(points, cam_pose, roll, pitch, yaw):
    ids = points[:,0:1]
    xyz = points[:,1:4]
    xyx_transformed = transform_nwu_to_camera(xyz, cam_pose, roll, pitch, yaw)
    return np.hstack([ids, xyx_transformed])

def transform_kps_camera2nwu(points, cam_pose, roll, pitch, yaw):
    ids = points[:,0:1]
    xyz = points[:,1:4]
    xyx_transformed = transform_camera_to_nwu(xyz, cam_pose, roll, pitch, yaw)
    return np.hstack([ids, xyx_transformed])

def transform_nwu_to_camera(points, cam_pose, roll, pitch, yaw):
    points_translated = points - cam_pose

    R1 = rotation_matrix_x(np.pi)
    Rpsi = rotation_matrix_z(yaw)
    Rtheta = rotation_matrix_y(pitch)
    Rphi = rotation_matrix_x(roll)
    R5 = rotation_matrix_x(np.pi/2)
    R6 = rotation_matrix_y(np.pi/2)

    R = R6 @ R5 @ Rphi @ Rtheta @ Rpsi @ R1
    points_out = points_translated @ R.T
    return points_out
    # return points_out - cam_pose

def transform_camera_to_nwu(points, cam_pose, roll, pitch, yaw):
    points_translated = points + cam_pose

    R1 = rotation_matrix_x(np.pi)
    Rpsi = rotation_matrix_z(yaw)
    Rtheta = rotation_matrix_y(pitch)
    Rphi = rotation_matrix_x(roll)
    R5 = rotation_matrix_x(np.pi/2)
    R6 = rotation_matrix_y(np.pi/2)

    R = R1 @ Rpsi @ Rtheta @ Rphi @ R5 @ R6
    # points_out = points @ R.T
    points_out = points_translated @ R.T
    return points_out


def rodrigues_rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    v1 = from_vec / (np.linalg.norm(from_vec) + 1e-12)
    v2 = to_vec / (np.linalg.norm(to_vec) + 1e-12)
    cross = np.cross(v1, v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        else:
            axis = np.array([1.0, 0.0, 0.0])
            if abs(v1[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - v1 * np.dot(axis, v1)
            axis /= (np.linalg.norm(axis) + 1e-12)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            return np.eye(3) + 2 * K @ K
    axis = cross / (np.linalg.norm(cross) + 1e-12)
    angle = np.arccos(dot)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

import open3d as o3d

def apply_transform_mesh(
    mesh: o3d.geometry.TriangleMesh,
    T: np.ndarray,
    *,
    inplace: bool = False
) -> o3d.geometry.TriangleMesh:
    """
    Apply a 4x4 homogeneous transform to an Open3D TriangleMesh.

    Convention: x' = R x + t, with R = T[:3,:3], t = T[:3,3].

    Args
    ----
    mesh : o3d.geometry.TriangleMesh
        Input mesh.
    T : (4,4) array-like
        Homogeneous transform.
    inplace : bool (default False)
        If True, modify the input mesh in-place and return it.
        If False, work on a copy and return the transformed copy.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Transformed mesh (same object if inplace=True).
    """
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("mesh must be an Open3D TriangleMesh.")

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be shape (4,4); got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]

    # choose target mesh
    out = mesh if inplace else (mesh.clone() if hasattr(mesh, "clone") else o3d.geometry.TriangleMesh(mesh))

    # vertices
    V = np.asarray(out.vertices, dtype=float)
    V[...] = V @ R.T + t  # x' = R x + t

    # vertex normals
    if out.has_vertex_normals():
        N = np.asarray(out.vertex_normals, dtype=float)
        N[...] = N @ R.T
        # renormalize (avoid divide-by-zero)
        nrm = np.linalg.norm(N, axis=1, keepdims=True)
        nz = nrm.squeeze(-1) > 0
        N[nz] /= nrm[nz]

    # triangle normals (if present)
    if out.has_triangle_normals():
        TN = np.asarray(out.triangle_normals, dtype=float)
        TN[...] = TN @ R.T
        nrm = np.linalg.norm(TN, axis=1, keepdims=True)
        nz = nrm.squeeze(-1) > 0
        TN[nz] /= nrm[nz]

    return out

def transform_in_xy_plane(corners: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    """
    Rotate + translate 3D corner points using aerospace yaw (psi).
    Convention: clockwise positive, rotation about Z.
    """
    if corners.shape != (4, 3):
        raise ValueError("corners must be of shape (4,3)")

    R_psi = np.array([
        [ np.cos(psi),  np.sin(psi), 0],
        [-np.sin(psi),  np.cos(psi), 0],
        [ 0,            0,           1]
    ], dtype=float)

    rotated = (R_psi @ corners.T).T
    translated = rotated + np.array([x, y, 0.0], dtype=float)
    return translated