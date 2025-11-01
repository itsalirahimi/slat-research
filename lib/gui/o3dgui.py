import threading
import open3d as o3d
import open3d.visualization.gui as gui
from projection.config import VisMode

class O3DGUI:
    def __init__(self, visMode):
        self.advance = threading.Event()
        self.scene_lock = threading.Lock()  
        self.visMode = visMode
        self.gui_thread = threading.Thread(target=self.do_gui, name="the-thread")
        self.gui_thread.start()

    def do_gui(self):
        if self.visMode == VisMode.Null:
            return  # No visualization, exit the method early

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("3D Mapper", 800, 600)
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)
        # Create the coordinate frame (three perpendicular arrows)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
        # Add the coordinate frame to the scene
        self.scene.scene.add_geometry("axis", axis, o3d.visualization.rendering.MaterialRecord())
        # Set the key event to trigger the 'project' method on 'A'
        self.window.set_on_key(self.on_key)
        gui.Application.instance.run()

    def _mat_points(self, size=4.0):
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = float(size)
        return m

    def on_key(self, e):
        if e.type != gui.KeyEvent.Type.DOWN:
            return False
        if e.key == gui.KeyName.A:
            print("user request: advance one frame")
            self.advance.set()
            return True
        elif e.key == gui.KeyName.Q:
            gui.Application.instance.quit(); 
            return True
        return False
