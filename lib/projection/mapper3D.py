import open3d as o3d
import numpy as np
from gui.o3dgui import O3DGUI
from .helper import *
from utils.o3dviz import mat_mesh, fit_camera
from .config import *
from utils.io import save_pcd
from utils.conversion import pcm2pcd

class Mapper3D(O3DGUI):
    def __init__(self, config: Mapper3DConfig):
        assert config.color_mode in ['image', 'proximity', 'constant', 'none'], \
            "color_mode must be 'image', 'proximity', 'constant', or 'none'"
        self.config = config
        self.it = 0
        os.makedirs(self.config.output_dir, exist_ok=True)
        super().__init__(self.config.visMode)

    def project(self, metric_depth, pose, cimg, filename, bgpcd=None):
        """
        Projects the 3D point cloud based on the provided metric depth and pose, and shows the visualization.
        """
        projected_pc, mc = project3D(metric_depth, pose, self.config.hfov_deg, 
                                    move=self.config.on_video,
                                    pyramidProj=self.config.pyramidProj, 
                                    scaling=self.config.scaling,
                                    do_rotate=True)
        if bgpcd is not None:
            if not self.config.in_camera:
                tmp = np.asarray(bgpcd.points) @ pose.getCAM2NWU().T
                bgpcd.points = o3d.utility.Vector3dVector(tmp)
            if mc is not None:
                bgpcd.points = o3d.utility.Vector3dVector(np.asarray(bgpcd.points) + mc)

        _pcd = pcm2pcd(projected_pc, cimg)

        # ======== Write to disk ========
        _filename = os.path.join(self.config.output_dir, f"{filename}.pcd")
        if self.config.do_save:
            save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=_filename)
            print(f"Wrote pcd file on {_filename}")

        # ======== Visualize ========
        with self.scene_lock:
            if self.config.visMode == VisMode.MSingle:
                name = f"points_{self.it}"
                mname = f"bgpcd_{self.it}"
                self.scene.scene.remove_geometry(name)
                self.scene.scene.remove_geometry(mname)
            elif self.config.visMode == VisMode.MAccum:
                pass
            self.it += 1
            name = f"points_{self.it}"
            mname = f"bgpcd_{self.it}"
            self.scene.scene.add_geometry(name, _pcd, self._mat_points(5.0))
            if bgpcd is not None:
                self.scene.scene.add_geometry(mname, bgpcd, mat_mesh())
            fit_camera(self.scene.scene, [_pcd])


