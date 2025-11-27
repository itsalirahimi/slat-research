import open3d as o3d
import numpy as np
from gui.o3dgui import O3DGUI
from .helper import *
from utils.o3dviz import mat_mesh, fit_camera
from .config import *
from ioHandle.IOHandler import save_pcd
from utils.conversion import pcm2pcd

class Mapper3D(O3DGUI):
    def __init__(self, config: Mapper3DConfig):
        assert config.color_mode in ['image', 'proximity', 'constant', 'none'], \
            "color_mode must be 'image', 'proximity', 'constant', or 'none'"
        self.config = config
        self.it = 0
        super().__init__(self.config.visMode)

    def project(self, metric_depth, pose, cimg, filename):
        """
        Projects the 3D point cloud based on the provided metric depth and pose, and shows the visualization.
        """
        H, W = metric_depth.shape
        ds_ratio = (self.config.downsample_pts / (H*W)) ** 0.5
        W_ds = int(ds_ratio*W)
        H_ds = int(ds_ratio*H)
        metric_depth_ds = cv2.resize(metric_depth, (W_ds, H_ds), interpolation=cv2.INTER_AREA)
        cimg_ds = cv2.resize(cimg, (W_ds, H_ds), interpolation=cv2.INTER_AREA)

        projected_pc, mc = project3D(metric_depth_ds, pose, self.config.hfov_deg, 
                                    move=self.config.on_video,
                                    pyramidProj=False, 
                                    do_rotate=True)
        
        projected_pc_can, mc = project3D(metric_depth_ds, pose, self.config.hfov_deg, 
                                    move=self.config.on_video,
                                    pyramidProj=True, 
                                    do_rotate=True)
        
        # projected_pc += np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]]).T
        # projected_pc_can += np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]]).T
        _pcd = pcm2pcd(projected_pc, cimg_ds)
        _pcd_c = pcm2pcd(projected_pc_can, cimg_ds)

        # ======== Write to disk ========
        _filename_radial = os.path.join(self.config.radial_dir, f"{filename}.pcd")
        _filename_canonical = os.path.join(self.config.canonical_dir, f"{filename}.pcd")
        _filename_img = os.path.join(self.config.rgb_dir, f"{filename}.png")
        if self.config.do_save:
            save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=_filename_radial)
            save_pcd(np.asarray(_pcd_c.points), np.asarray(_pcd_c.colors), filepath=_filename_canonical)
            cv2.imwrite(_filename_img, cimg_ds)
            print(f"Wrote pcd file on {_filename_radial}")

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
            fit_camera(self.scene.scene, [_pcd])
