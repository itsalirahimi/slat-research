"""Optional key-driven cumulative-map viewer."""

import threading
from typing import Optional

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:  # Headless mapping remains importable without Open3D.
    o3d = None
    gui = None
    rendering = None


class MappingVisualizer:
    """Show the cumulative cloud and release one frame for each ``A`` key."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.advance = threading.Event()
        self.closed = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.window = None
        self.scene = None
        self._camera_initialized = False
        self._thread = None

        if not self.enabled:
            self.ready.set()
            return
        if o3d is None or gui is None:
            raise RuntimeError("Open3D GUI support is required for --visualize")
        self._thread = threading.Thread(
            target=self._run,
            name="mapping-visualizer",
            daemon=False,
        )
        self._thread.start()
        if not self.ready.wait(timeout=15.0):
            raise RuntimeError("mapping visualizer did not initialize")
        if self.error is not None:
            raise RuntimeError(
                "mapping visualizer failed to initialize; omit --visualize on a "
                "headless machine"
            ) from self.error

    def _run(self):
        try:
            gui.Application.instance.initialize()
            self.window = gui.Application.instance.create_window(
                "Incremental map — press A to advance, Q to quit", 1000, 700
            )
            self.scene = gui.SceneWidget()
            self.scene.scene = rendering.Open3DScene(self.window.renderer)
            self.window.add_child(self.scene)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=5.0, origin=[0.0, 0.0, 0.0]
            )
            self.scene.scene.add_geometry("axes", axis, rendering.MaterialRecord())
            self.window.set_on_key(self._on_key)
            self.window.set_on_close(self._on_close)
            self.ready.set()
            gui.Application.instance.run()
        except BaseException as exc:  # Propagate initialization failure to caller.
            self.error = exc
            self.ready.set()
        finally:
            self.closed.set()
            self.advance.set()

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        if event.key == gui.KeyName.A:
            self.advance.set()
            return True
        if event.key == gui.KeyName.Q:
            self.closed.set()
            self.advance.set()
            gui.Application.instance.quit()
            return True
        return False

    def _on_close(self):
        self.closed.set()
        self.advance.set()
        return True

    def wait_for_advance(self, index: int, name: str) -> bool:
        if not self.enabled:
            return True
        print(f"[mapping][viewer] press A to process frame {index}: {name}; Q quits")
        while not self.closed.is_set():
            if self.advance.wait(timeout=0.10):
                self.advance.clear()
                return not self.closed.is_set()
        return False

    def update(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        status: str,
    ):
        if not self.enabled or self.closed.is_set():
            return
        points = np.asarray(points, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.float64)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        if colors.shape == points.shape:
            cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0

        def apply_update():
            if self.closed.is_set() or self.scene is None:
                return
            if self.scene.scene.has_geometry("cumulative_map"):
                self.scene.scene.remove_geometry("cumulative_map")
            if len(points):
                self.scene.scene.add_geometry("cumulative_map", cloud, material)
                if not self._camera_initialized:
                    bounds = cloud.get_axis_aligned_bounding_box()
                    if np.any(bounds.get_extent() > 0):
                        self.scene.setup_camera(60.0, bounds, bounds.get_center())
                        self._camera_initialized = True
            self.window.title = f"Incremental map — {status} — A: next, Q: quit"

        gui.Application.instance.post_to_main_thread(self.window, apply_update)

    def finish(self):
        if not self.enabled:
            return
        if not self.closed.is_set():
            print("[mapping][viewer] sequence complete; press Q to close")
        if self._thread is not None:
            self._thread.join()

    def close(self):
        """Close the viewer without waiting, used when mapping raises an error."""
        if not self.enabled:
            return
        self.closed.set()
        self.advance.set()
        if self.window is not None:
            try:
                gui.Application.instance.post_to_main_thread(
                    self.window, gui.Application.instance.quit
                )
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
