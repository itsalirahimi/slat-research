
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import open3d as o3d

def mat_points(size=4.0):
    m = rendering.MaterialRecord()
    m.shader = "defaultUnlit"
    m.point_size = float(size)
    return m

def mat_mesh():
    m = rendering.MaterialRecord()
    m.shader = "defaultLit"
    m.base_color = (0.7, 0.7, 0.9, 1.0)
    m.base_roughness = 0.8
    return m

def mat_mesh_tinted(rgba):
    m = rendering.MaterialRecord(); m.shader = "defaultLit"
    m.base_color = tuple(float(c) for c in rgba); m.base_roughness = 0.8; return m

def fit_camera(scene, geoms):
    aabbs = [g.get_axis_aligned_bounding_box() for g in geoms if g is not None]
    if not aabbs:
        return
    mins = np.min([a.min_bound for a in aabbs], axis=0)
    maxs = np.max([a.max_bound for a in aabbs], axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(max(maxs - mins))
    if extent <= 0:
        extent = 1.0
    eye = center + np.array([0, -3.0 * extent, 1.8 * extent])
    up = np.array([0, 0, 1])
    scene.camera.look_at(center, eye, up)


def visualize_spline_mesh(ctrl_pcd, surf_mesh, ext_pcd, name, proj_pcd=None):
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(name, 1024, 768)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name=name, width=1440, height=900)

    if ctrl_pcd is not None:
        scene.scene.add_geometry("ctrl_pcd", ctrl_pcd, mat_points(8.0))
    if surf_mesh is not None:
        scene.scene.add_geometry("surf_mesh", surf_mesh, mat_mesh())
    if ext_pcd is not None:
        scene.scene.add_geometry("ext_pcd", ext_pcd, mat_points(2.0))
    if proj_pcd is not None:
        scene.scene.add_geometry("proj_pcd", proj_pcd, mat_points(2.0))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    scene.scene.add_geometry("axis", axis, mat_mesh())

    if proj_pcd is None:
        fit_camera(scene.scene, [ctrl_pcd, surf_mesh, ext_pcd, axis])
    else:
        fit_camera(scene.scene, [ctrl_pcd, surf_mesh, ext_pcd, proj_pcd, axis])
    gui.Application.instance.run()
    # vis.run()
    # vis.destroy_window()

def make_lineset_for_pairs(src_pts: np.ndarray, dst_pts: np.ndarray, color=(1.0, 0.0, 1.0)) -> o3d.geometry.LineSet:
    """Build a LineSet connecting each src point to its corresponding dst point."""
    if src_pts.shape != dst_pts.shape:
        raise ValueError("Source and destination point arrays must have the same shape.")
    n = src_pts.shape[0]
    if n == 0:
        return o3d.geometry.LineSet()
    points = np.vstack([src_pts, dst_pts])
    lines = np.column_stack([np.arange(n, dtype=np.int32), np.arange(n, 2*n, dtype=np.int32)])
    colors = np.tile(np.asarray(color, float)[None, :], (n, 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(float))
    ls.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    ls.colors = o3d.utility.Vector3dVector(colors.astype(float))
    return ls


def estimate_marker_radius(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> float:
    """Heuristic sphere radius based on scene scale."""
    verts = np.asarray(mesh.vertices)
    all_pts = verts if points.size == 0 else np.vstack([verts, points])
    if all_pts.shape[0] < 2:
        return 1e-2
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    diag = np.linalg.norm(mx - mn)
    if not np.isfinite(diag) or diag <= 0:
        return 1e-2
    return float(0.01 * diag)  # 1% of scene diagonal

def viz_pcd(pcd, title="Point Cloud", point_size=3, bg_color=(0, 0, 0)):
    if pcd.is_empty():
        raise ValueError("pcd is empty.")
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    try:
        # Newer API (Open3D ≥ 0.17)
        o3d.visualization.draw(
            [
                {"name": "pcd", "geometry": pcd, "material": {"point_size": point_size}},
                {"name": "axes", "geometry": frame},
            ],
            title=title,
            bg_color=bg_color,
        )
    except Exception:
        # Fallback for older Open3D
        o3d.visualization.draw_geometries([pcd, frame], window_name=title)

