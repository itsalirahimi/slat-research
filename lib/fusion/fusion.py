import os
from typing import Iterable, Tuple
import numpy as np
import open3d as o3d
from geom.surfaces import bspline_surface_mesh_from_ctrl
from gui.o3dgui import O3DGUI
from ioHandle.IOHandler import load_pcd, save_pcd, save_pcm_as_pcd
from utils.o3dviz import fit_camera
from .config import BGPatternFuserConfig, FlatFusionMode
from .helper import build_camera_rays, calc_ratio_map, fit_ctrl_grid_from_point_cloud, intersect_rays_with_spline, unfold_depth, drop_depth, NDFDrop_depth, calc_ground_depth, \
    depyramidize_pointCloud
from projection.helper import project3D, NULL_SCALE_MIN_Z, computeGeps, scale_pcm
from utils.conversion import pcd2pcm, pcdArr2pcd, pcd2hw1, pcm2pcd, pcm2pcdArr, pcdArr2pcm
from projection.config import VisMode
import open3d.visualization.rendering as rendering


class BGPatternFuser(O3DGUI):
    def __init__(self, config: BGPatternFuserConfig):
        self.config = config
        self.it = 0
        self.shape = None
        self.projected_pcd = None
        super().__init__(self.config.visMode)

    def set_internal_shape(self, shape):
        if self.shape is None:
            self.shape = shape[:2]
        else:
            assert self.shape == shape[:2], "Inconsistent image shape detected in fuser"

    def fuse(self, cimg, prj_pcd_rad, prj_pcd_can, bg_pcd_rad, bg_pcd_can, pose, filename):
        H, W, _ = cimg.shape
        ratio_map, gep, _ = calc_ratio_map(bg_pcd_can, pose, H, W, self.config.hfov_deg)
        ratio_map = ratio_map[..., np.newaxis]      # (H, W, 1)
        prj_pcm_can = pcd2pcm(prj_pcd_can, H, W)
        tilt_biased_bg = prj_pcm_can * ratio_map

        # bg_rad_ctrl = fit_ctrl_grid_from_point_cloud(
        #     bg_pcd_rad, grid_w=20, grid_h=20, k_neighbors=10
        # )
        # bg_rad_mesh = bspline_surface_mesh_from_ctrl(
        #     bg_rad_ctrl, grid_w=20, grid_h=20, su=40, sv=40
        # )
        # origins, dirs = build_camera_rays(
        #         img_h=H,
        #         img_w=W,
        #         hfov_deg=62,
        #         pose=pose,
        #         pyramidProj=False
        #     )

        # _, p_bg_rad_mesh = intersect_rays_with_spline(bg_rad_mesh, origins, dirs)
        # p_bg_rad_mesh_pcd = pcdArr2pcd(p_bg_rad_mesh)
        # bgcpm = pcd2pcm(p_bg_rad_mesh_pcd, H, W)
        # tilt_biased_bg_rad = bgcpm * (ratio_map[..., np.newaxis])
        # tilt_biased_bg_rad_pcd = pcm2pcd(tilt_biased_bg_rad)
        # # save_pcd(np.asarray(bpcd.points), np.asarray(np.zeros_like(bpcd.points)), f"bpcd.pcd")

        # prj_pcm_rad = pcd2pcm(prj_pcd_rad, H, W)
        # tilt_biased_prj_rad = prj_pcm_rad * (ratio_map[..., np.newaxis]) 
        # fixed_fg = unfold_depth(tilt_biased_prj_rad, tilt_biased_bg_rad_pcd, H, W)

        fused_pcm_nwu = tilt_biased_bg
        bg_pcm_can = pcd2pcm(bg_pcd_can, H, W)
        fused_pcm_nwu = scale_pcm(fused_pcm_nwu, np.nanmin(bg_pcm_can[:,:,2]), -pose.p6.z)
        gep_pcm = pcdArr2pcm(gep, H, W)
        gep = scale_pcm(gep, np.mean(gep_pcm[:,:,2]), -pose.p6.z)


        _pcd = pcm2pcd(fused_pcm_nwu, cimg)
        _pcd_gep = pcdArr2pcd(gep)
        _gep_d_mhw1 = pcd2hw1(_pcd_gep, H, W)
        _gep_p_mhw3 = pcd2pcm(_pcd_gep, H, W)
        _mfused = pcd2hw1(_pcd, H, W)

        # ======== Write to disk ========
        _filename_fuse = os.path.join(self.config.fusion_dir, f"{filename}.pcd")
        _filename_gep = os.path.join(self.config.gep_dir, f"{filename}.pcd")
        _filename_mfused = os.path.join(self.config.mfused_dir, f"{filename}.npy")
        _filename_mgepd = os.path.join(self.config.mgepd_dir, f"{filename}.npy")
        _filename_mgepp = os.path.join(self.config.mgepp_dir, f"{filename}.npy")
        if self.config.do_save:
            save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=_filename_fuse)
            save_pcd(np.asarray(_pcd_gep.points), np.asarray(_pcd.colors), filepath=_filename_gep)
            np.save(_filename_mfused, _mfused)
            np.save(_filename_mgepd, _gep_d_mhw1)
            np.save(_filename_mgepp, _gep_p_mhw3)

            print(f"Wrote pcd file on {_filename_fuse}")

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

            # self.scene.scene.add_geometry(name, _pcd, self._mat_points(5.0))
            self._safe_add_pcd(name, _pcd, point_size=5.0)
            fit_camera(self.scene.scene, [_pcd])

    def _safe_add_pcd(self, name, pcd, point_size=5.0):
        # 1) Make sure name is str
        name = str(name)

        # 2) Make sure we have legacy PointCloud (not tensor)
        if hasattr(o3d, "t") and isinstance(pcd, o3d.t.geometry.PointCloud):
            pcd = pcd.to_legacy()

        # 3) Extract and clean points
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            print(f"[SAFE_ADD] '{name}' has no points → skip add_geometry")
            return

        # Remove NaN/Inf
        mask = np.isfinite(pts).all(axis=1)
        if not np.all(mask):
            print(f"[SAFE_ADD] '{name}' removing {np.count_nonzero(~mask)} non-finite points")
            pts = pts[mask]
            if pts.size == 0:
                print(f"[SAFE_ADD] '{name}' no finite points left → skip")
                return
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # 4) Check AABB
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()  # np.array([ex, ey, ez])
        print(f"[SAFE_ADD] '{name}' extent:", extent)

        # Filament treats an empty / zero-extent box as error
        if np.all(extent <= 0):
            print(f"[SAFE_ADD] '{name}' has empty AABB → skip add_geometry")
            return

        # Optional: if it's *almost* zero in all dims, slightly jitter 1–2 points
        eps = 1e-9
        if np.all(extent < eps):
            print(f"[SAFE_ADD] '{name}' has degenerate AABB, jittering a couple of points")
            pts = np.asarray(pcd.points)
            if pts.shape[0] >= 2:
                pts[0] += np.array([1e-3, 0.0, 0.0])
                pts[1] += np.array([0.0, 1e-3, 0.0])
                pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
                bbox = pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                print(f"[SAFE_ADD] '{name}' new extent:", extent)
            else:
                print(f"[SAFE_ADD] '{name}' has <2 pts and degenerate AABB → skip")
                return

        # 5) Material
        mat = self._mat_points(float(point_size))
        assert isinstance(mat, rendering.MaterialRecord)

        # 6) Replace if already exists
        if self.scene.scene.has_geometry(name):
            self.scene.scene.remove_geometry(name)

        # 7) Finally call add_geometry
        self.scene.scene.add_geometry(name, pcd, mat)


    # def fuse(self, cimg, rd_pcd_rad, bg_pcd_rad, pose, filename):
    #     H, W, _ = cimg.shape
    #     rd_pcm_nwu = pcd2pcm(rd_pcd_rad, H, W)
    #     self.set_internal_shape(rd_pcm_nwu.shape)
    #     gep_pcm_nwu_nonscaled = computeGeps(self.shape, self.config.hfov_deg, pose)
    #     gep_pcm_nwu = gep_pcm_nwu_nonscaled * calc_scale_factor(np.min(rd_pcm_nwu[:,:,2]), 
    #         scaling=Scaling.MIN_Z, bgz=None, pc_to_be_rescaled=gep_pcm_nwu_nonscaled)
        

    #     bg_rad_ctrl_flat = fit_ctrl_grid_from_point_cloud(
    #         bg_pcd_rad, grid_w=20, grid_h=20, k_neighbors=10
    #     )
        
    #     if self.config.flat_mode == FlatFusionMode.NDFDrop:
    #         fused_pcm_nwu, base_elevs = NDFDrop_depth(rd_pcm_nwu, bg_rad_ctrl_flat, gep_pcm_nwu, H, W)

    #     elif self.config.flat_mode == FlatFusionMode.Unfold:
    #         spline_mesh = bspline_surface_mesh_from_ctrl(
    #             bg_rad_ctrl_flat, grid_w=20, grid_h=20, su=200, sv=200
    #         )

    #         origins, dirs = build_camera_rays(
    #                 img_h=H,
    #                 img_w=W,
    #                 hfov_deg=62,
    #                 pose=pose,
    #                 pyramidProj=False
    #             )
    #         t_mesh = intersect_rays_with_spline(spline_mesh, origins, dirs)
    #         mesh_hit_mask = np.isfinite(t_mesh) & (t_mesh > 0.0)
    #         mesh_pts = origins[mesh_hit_mask] + dirs[mesh_hit_mask] * t_mesh[mesh_hit_mask, None]
    #         bg_pcd_nwu = pcdArr2pcd(mesh_pts)
    #         fused_pcm_nwu = unfold_depth(rd_pcm_nwu, bg_pcd_nwu, bg_rad_ctrl_flat, gep_pcm_nwu, H, W)

    #     else:
    #         raise ValueError("Unknown refusion mode")

    #     # save_pcm_as_pcd(fused_pcm_nwu_scaled, "fused_pcm_nwu_scaled.pcd", color = [255, 0, 0])
    #     # def resacle_and_repose(pcd):
    #     #     pcd *= 113.86/NULL_SCALE_MIN_Z
    #     #     pcd[:,:, 0] += 65
    #     #     pcd[:,:, 1] += 21
    #     #     return pcd
    #     # iterlist = []
    #     # a = gep_pcm_nwu[0,0,0]
    #     # b = gep_pcm_nwu[0,0,1]
    #     # iterlist.append((a,b))
    #     # a = gep_pcm_nwu[0,W-1,0]
    #     # b = gep_pcm_nwu[0,W-1,1]
    #     # iterlist.append((a,b))
    #     # a = gep_pcm_nwu[H-1,W-1,0]
    #     # b = gep_pcm_nwu[H-1,W-1,1]
    #     # iterlist.append((a,b))
    #     # a = gep_pcm_nwu[H-1,0,0]
    #     # b = gep_pcm_nwu[H-1,0,1]
    #     # iterlist.append((a,b))
    #     # fused_pcm_nwu_polygon = reshape_in_polygon(iterlist, fused_pcd_nwu, H, W)
    #     # save_pcd(np.asarray(fused_pcm_nwu_polygon.points),
    #     #             np.asarray(np.zeros_like(fused_pcm_nwu_polygon.points)),
    #     #             "fused_pcm_nwu_polygon.pcd")

    #     fused_pcm_nwu = fused_pcm_nwu * pose.p6.z/NULL_SCALE_MIN_Z
        
    #     _pcd = pcm2pcd(fused_pcm_nwu, cimg)
    #     # ======== Write to disk ========
    #     _filename_fuse = os.path.join(self.config.fusion_dir, f"{filename}.pcd")
    #     _filename_gep = os.path.join(self.config.gep_dir, f"{filename}.pcd")
    #     if self.config.do_save:
    #         save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=_filename_fuse)
    #         _pcd_gep = pcm2pcd(gep_pcm_nwu, cimg)
    #         save_pcd(np.asarray(_pcd_gep.points), np.asarray(_pcd_gep.colors), filepath=_filename_gep)
    #         print(f"Wrote pcd file on {_filename_fuse}")

    #     # ======== Visualize ========
    #     with self.scene_lock:
    #         if self.config.visMode == VisMode.MSingle:
    #             name = f"points_{self.it}"
    #             mname = f"bgpcd_{self.it}"
    #             self.scene.scene.remove_geometry(name)
    #             self.scene.scene.remove_geometry(mname)
    #         elif self.config.visMode == VisMode.MAccum:
    #             pass
    #         self.it += 1
    #         name = f"points_{self.it}"
    #         mname = f"bgpcd_{self.it}"
    #         self.scene.scene.add_geometry(name, _pcd, self._mat_points(5.0))
    #         fit_camera(self.scene.scene, [_pcd])

        
    # def fuse_flat_ground(self, cimg, data, bg_pcd, pose, filename):
    #     H, W, _ = cimg.shape
        
    #     if self.config.flat_mode == FlatFusionMode.Depyramidize:
    #         # 'Depyramidize' fusion method requires pyramid projection math
    #         rd_pcm_cam, _ = pcd2pcm(data, H, W)
    #         rd_pcm_nwu = None
    #         self.set_internal_shape(rd_pcm_cam.shape)
    #         rdpcarr_cam = pcm2pcdArr(rd_pcm_cam)
    #         fus_pcm_arr_cam, _, _ = depyramidize_pointCloud(rdpcarr_cam)
    #         fused_pcm_cam = pcdArr2pcm(fus_pcm_arr_cam, rd_pcm_cam.shape[0], rd_pcm_cam.shape[1])
    #         fused_pcm_cam *= calc_scale_factor(pose.p6.z, Scaling.MEAN_Z, 
    #                                       bgz=np.ones(self.shape)*NULL_SCALE_MIN_Z)
    #         fused_pcm_nwu = fused_pcm_cam @ pose.getCAM2NWU().T

    #     else:
    #         rd_pcm_cam = None
    #         rd_pcm_nwu = pcd2pcm(data, H, W)
    #         self.set_internal_shape(rd_pcm_nwu.shape)

    #         gep_pcm_nwu_nonscaled = computeGeps(self.shape, 70, pose)
    #         gep_pcm_nwu = gep_pcm_nwu_nonscaled * calc_scale_factor(np.min(rd_pcm_nwu[:,:,2]), 
    #             scaling=Scaling.MIN_Z, bgz=None, pc_to_be_rescaled=gep_pcm_nwu_nonscaled)
            
    #         if self.config.flat_mode == FlatFusionMode.NDFDrop:
    #             assert bg_pcd is not None, "'NDFDrop' fusion mode requires background data"

    #             bg_pcd_nwu = bg_pcd
    #             fused_pcm_nwu, base_elevs = NDFDrop_depth(rd_pcm_nwu, bg_pcd_nwu, gep_pcm_nwu)


    #         elif self.config.flat_mode == FlatFusionMode.Unfold:
    #             assert bg_pcd is not None, "'Unfold' fusion mode requires background data"

    #             bg_pcd_nwu = bg_pcd
    #             fused_pcm_nwu = unfold_depth(rd_pcm_nwu, bg_pcd_nwu, gep_pcm_nwu, H, W)

    #         else:
    #             raise ValueError("Unknown refusion mode")
            

    #         def resacle_and_repose(pcd):
    #             pcd *= 113.86/NULL_SCALE_MIN_Z
    #             pcd[:,:, 0] += 65
    #             pcd[:,:, 1] += 21
    #             return pcd
                
    #         save_debug_files = False
    #         if save_debug_files:
    #             save_pcm_as_pcd(fused_pcm_nwu, "fused_pcm_nwu.pcd", vizimg=cimg)

    #             fused_pcm_nwu_scaled = resacle_and_repose(fused_pcm_nwu)
    #             save_pcm_as_pcd(fused_pcm_nwu_scaled, "fused_pcm_nwu_scaled.pcd", color = [255, 0, 0])
                
    #             fused_pcm_cam = fused_pcm_nwu @ pose.getNWU2CAM()
    #             save_pcm_as_pcd(fused_pcm_cam, "fused_pcm_cam.pcd", color = [255, 0, 0])
                
    #             save_pcm_as_pcd(gep_pcm_nwu, "gep_pcm_nwu.pcd", color = [255, 255, 0])
    #             gep_pcm_nwu_scaled = resacle_and_repose(gep_pcm_nwu)
    #             save_pcm_as_pcd(gep_pcm_nwu_scaled, "gep_pcm_nwu_scaled.pcd", color = [255, 255, 0])

    #             rd_pcm_nwu = resacle_and_repose(rd_pcm_nwu)
    #             save_pcm_as_pcd(rd_pcm_nwu, "rd_pcm_nwu.pcd", color = [0, 255, 255])

    #             save_pcd(np.asarray(bg_pcd_nwu.points), np.asarray(np.zeros_like(bg_pcd_nwu.points)), "bg_pcd_nwu.pcd")

    #             fused_pcd_nwu = pcm2pcd(fused_pcm_nwu)
    #             iterlist = []
    #             a = gep_pcm_nwu[0,0,0]
    #             b = gep_pcm_nwu[0,0,1]
    #             iterlist.append((a,b))
    #             a = gep_pcm_nwu[0,W-1,0]
    #             b = gep_pcm_nwu[0,W-1,1]
    #             iterlist.append((a,b))
    #             a = gep_pcm_nwu[H-1,W-1,0]
    #             b = gep_pcm_nwu[H-1,W-1,1]
    #             iterlist.append((a,b))
    #             a = gep_pcm_nwu[H-1,0,0]
    #             b = gep_pcm_nwu[H-1,0,1]
    #             iterlist.append((a,b))
    #             fused_pcm_nwu_polygon = reshape_in_polygon(iterlist, fused_pcd_nwu,H, W)
    #             save_pcd(np.asarray(fused_pcm_nwu_polygon.points),
    #                      np.asarray(np.zeros_like(fused_pcm_nwu_polygon.points)),
    #                      "fused_pcm_nwu_polygon.pcd")
                
        
    #     # if self.config.on_video:
    #     #     fused_pcm_nwu += np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]]).T
        
    #     _pcd = pcm2pcd(fused_pcm_nwu, cimg)
    #     # ======== Write to disk ========
    #     _filename_fuse = os.path.join(self.config.fusion_dir, f"{filename}.pcd")
    #     _filename_gep = os.path.join(self.config.gep_dir, f"{filename}.pcd")
    #     if self.config.do_save:
    #         save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=_filename_fuse)
    #         _pcd_gep = pcm2pcd(gep_pcm_nwu, cimg)
    #         save_pcd(np.asarray(_pcd_gep.points), np.asarray(_pcd_gep.colors), filepath=_filename_gep)
    #         print(f"Wrote pcd file on {_filename_fuse}")

    #     # ======== Visualize ========
    #     with self.scene_lock:
    #         if self.config.visMode == VisMode.MSingle:
    #             name = f"points_{self.it}"
    #             mname = f"bgpcd_{self.it}"
    #             self.scene.scene.remove_geometry(name)
    #             self.scene.scene.remove_geometry(mname)
    #         elif self.config.visMode == VisMode.MAccum:
    #             pass
    #         self.it += 1
    #         name = f"points_{self.it}"
    #         mname = f"bgpcd_{self.it}"
    #         self.scene.scene.add_geometry(name, _pcd, self._mat_points(5.0))
    #         fit_camera(self.scene.scene, [_pcd])

    # def fuse_elevation(self):
    #     assert False, "Not yet developed"