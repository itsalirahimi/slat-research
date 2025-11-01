import os
import numpy as np

from gui.o3dgui import O3DGUI
from utils.io import save_pcd, save_pcm_as_pcd
from utils.o3dviz import fit_camera, mat_mesh, viz_pcd
from .config import BGPatternFuserConfig, FlatFusionMode
from .helper import unfold_depth, drop_depth, NDFDrop_depth, calc_ground_depth, \
    depyramidize_pointCloud, downsample_pcm
from projection.helper import project3D, calc_scale_factor, NULL_SCALE_MIN_Z, computeGeps, resize_keep_ar
from geom.rectification import rectify_xy_proj
from utils.conversion import pcd2pcm, pcdArr2pcd, pcm2pcd, pcm2pcdArr, pcdArr2pcm
from projection.config import Scaling, VisMode


class BGPatternFuser(O3DGUI):
    def __init__(self, config: BGPatternFuserConfig):
        self.config = config
        self.it = 0
        self.shape = None
        self.projected_pcd = None
        os.makedirs(self.config.output_dir, exist_ok=True)
        assert not self.config.in_camera, "When requesting fusion, the in_camera must be False (i.e. cam2nwu rotation required)"
        super().__init__(self.config.visMode)

    def set_internal_shape(self, shape):
        if self.shape is None:
            self.shape = shape[:2]
        else:
            assert self.shape == shape[:2], "Inconsistent image shape detected in fuser"

    def fuse_flat_ground(self, cimg, data, bg_pcd, pose, filename):
        H, W, _ = cimg.shape
        cimg = resize_keep_ar(cimg, self.config.downsample_dstW)

        if self.config.flat_mode == FlatFusionMode.Depyramidize:
            # 'Depyramidize' fusion method requires pyramid projection math
            rd_pcm_cam, _ = pcd2pcm(data, H, W)
            rd_pcm_nwu = None
            rd_pcm_cam = downsample_pcm(rd_pcm_cam, self.config.downsample_dstW)
            self.set_internal_shape(rd_pcm_cam.shape)
            rdpcarr_cam = pcm2pcdArr(rd_pcm_cam)
            fus_pcm_arr_cam, _, _ = depyramidize_pointCloud(rdpcarr_cam)
            fused_pcm_cam = pcdArr2pcm(fus_pcm_arr_cam, rd_pcm_cam.shape[0], rd_pcm_cam.shape[1])
            fused_pcm_cam *= calc_scale_factor(pose.p6.z, Scaling.MEAN_Z, 
                                          bgz=np.ones(self.shape)*NULL_SCALE_MIN_Z)
            fused_pcm_nwu = fused_pcm_cam @ pose.getCAM2NWU().T

        else:
            rd_pcm_cam = None
            rd_pcm_nwu = pcd2pcm(data, H, W)
            rd_pcm_nwu = downsample_pcm(rd_pcm_nwu, self.config.downsample_dstW)
            self.set_internal_shape(rd_pcm_nwu.shape)
            gep_pcm_nwu_nonscaled = computeGeps(self.shape, 63, pose)
            gep_pcm_nwu = gep_pcm_nwu_nonscaled * calc_scale_factor(np.min(rd_pcm_nwu[:,:,2]), 
                scaling=Scaling.MIN_Z, bgz=None, pc_to_be_rescaled=gep_pcm_nwu_nonscaled)
            # save_pcm_as_pcd(gep_pcm_nwu, "gep_pcm_nwu.pcd", color = [255, 0, 0])
            if self.config.flat_mode == FlatFusionMode.NDFDrop:
                assert bg_pcd is not None, "'NDFDrop' fusion mode requires background data"
                # bg_pcdArr_nwu = np.asarray(bg_pcd.points) @ pose.getCAM2NWU().T
                bg_pcdArr_nwu = np.asarray(bg_pcd.points)
                bg_pcd_nwu = pcdArr2pcd(bg_pcdArr_nwu)
                fused_pcm_nwu, base_elevs = NDFDrop_depth(rd_pcm_nwu, bg_pcd_nwu, gep_pcm_nwu)

                # saving debug pcd
                # save_pcm_as_pcd(fused_pcm_nwu, "fused_pcm_nwu1.pcd", vizimg=cimg)
                # fused_pcm_nwu *= 113.86/NULL_SCALE_MIN_Z
                # fused_pcm_nwu[:,:, 0] += 65
                # fused_pcm_nwu[:,:, 1] += 21
                # save_pcm_as_pcd(fused_pcm_nwu, "fused_pcm_nwu2.pcd", color = [255, 0, 0])
                
                # fused_pcm_cam = fused_pcm_nwu @ pose.getNWU2CAM()
                # save_pcm_as_pcd(fused_pcm_cam, "fused_pcm_cam1.pcd", color = [255, 0, 0])
                
                # fused_pcm_cam = rectify_xy_proj(fused_pcm_cam)
                # save_pcm_as_pcd(fused_pcm_cam, "fused_pcm_cam2.pcd", color = [255, 0, 0])

                # fused_pcm_nwu = rectify_xy_proj(fused_pcm_nwu)
                # save_pcm_as_pcd(fused_pcm_nwu, "fused_pcm_nwu2.pcd", color = [255, 0, 0])

                # fused_pcm_nwu = fused_pcm_cam @ pose.getCAM2NWU()
                # save_pcm_as_pcd(fused_pcm_nwu, "fused_pcm_nwu3.pcd", color = [255, 0, 0])
                # gep_pcm_nwu *= 113.86/NULL_SCALE_MIN_Z
                # gep_pcm_nwu[:,:, 0] += 65
                # gep_pcm_nwu[:,:, 1] += 21
                # save_pcm_as_pcd(gep_pcm_nwu, "gep_pcm_nwu.pcd", color = [255, 255, 0])
                # rd_pcm_nwu *= 113.86/NULL_SCALE_MIN_Z
                # rd_pcm_nwu[:,:, 0] += 65
                # rd_pcm_nwu[:,:, 1] += 21
                # save_pcm_as_pcd(rd_pcm_nwu, "rd_pcm_nwu.pcd", color = [0, 255, 255])
                # save_pcd(np.asarray(bg_pcd_nwu.points), np.asarray(np.zeros_like(bg_pcd_nwu.points)), "bg_pcd_nwu.pcd")
                # fused_pcm_nwu *= calc_scale_factor(-abs(pose.p6.z), Scaling.MEAN_Z, 
                #                             bgz=np.ones(self.shape)*NULL_SCALE_MIN_Z)
                # save_pcm_as_pcd(fused_pcm_cam, "fused_pcm_cam.pcd", color = [255, 0, 0])
                # save_pcm_as_pcd(fused_pcm_nwu, "fused_pcm_nwu.pcd", color = [255, 0, 0])

            elif self.config.flat_mode == FlatFusionMode.Unfold:
                assert bg_pcd is not None, "'Unfold' fusion mode requires background data"
                assert False, "Unfold method not yet tested"

            else:
                raise ValueError("Unknown refusion mode")
        
        # if self.config.on_video:
        #     fused_pcm_nwu += np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]]).T
        
        _pcd = pcm2pcd(fused_pcm_nwu, cimg)
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
            self.scene.scene.add_geometry(name, _pcd, self._mat_points(1.0))
            fit_camera(self.scene.scene, [_pcd])

    def fuse_elevation(self):
        assert False, "Not yet developed"