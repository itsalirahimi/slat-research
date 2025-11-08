import os
from typing import Iterable, Tuple
import numpy as np
import open3d as o3d
from gui.o3dgui import O3DGUI
from ioHandle.IOHandler import load_pcd, save_pcd, save_pcm_as_pcd
from kinematics.clouds import apply_transform_points, orient_point_cloud_cgplane_global
from utils.o3dviz import fit_camera, mat_mesh, viz_pcd
from .config import BGPatternFuserConfig, FlatFusionMode
from .helper import unfold_depth, drop_depth, NDFDrop_depth, calc_ground_depth, \
    depyramidize_pointCloud, downsample_pcm
from projection.helper import project3D, calc_scale_factor, NULL_SCALE_MIN_Z, computeGeps, resize_keep_ar
from geom.rectification import rectify_xy_proj
from utils.conversion import pcd2pcm, pcdArr2pcd, pcm2pcd, pcm2pcdArr, pcdArr2pcm
from projection.config import Scaling, VisMode


def _pcd2xyz_grid(pcd: o3d.geometry.PointCloud, H: int, W: int) -> np.ndarray:
    """Convert an Open3D point cloud with H*W points (row-major) into an H×W×3 array."""
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] != H * W:
        raise ValueError(f"Point cloud has {pts.shape[0]} points, expected {H*W}.")
    return pts.reshape(H, W, 3)

def _xyz_grid2pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert an H×W×3 array back to an Open3D point cloud."""
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(xyz.reshape(-1, 3))
    return out

def _polygon_signed_area(poly_xy: np.ndarray) -> float:
    """Twice the signed area (positive=CCW) of a closed polygon given as (N,2)."""
    s = 0.0
    for i in range(len(poly_xy)):
        x1, y1 = poly_xy[i]
        x2, y2 = poly_xy[(i + 1) % len(poly_xy)]
        s += x1 * y2 - x2 * y1
    return s

def _order_quad_tl_tr_br_bl(pts4_xy: np.ndarray) -> np.ndarray:
    """
    Order 4 XY points robustly to [TL, TR, BR, BL].
    Steps:
      1) Sort CCW around centroid
      2) Rotate so the first is TL (min y, then min x)
      3) Ensure clockwise order (TL,TR,BR,BL)
    """
    if pts4_xy.shape != (4, 2):
        raise ValueError("four_points must be (4,2) array-like of XY.")
    P = np.asarray(pts4_xy, dtype=np.float64)

    c = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    P_ccw = P[np.argsort(ang)]  # CCW around centroid

    # make first = TL by (y,x)
    tl_idx = np.lexsort((P_ccw[:, 0], P_ccw[:, 1]))[0]
    P_ccw = np.roll(P_ccw, -tl_idx, axis=0)

    # ensure clockwise order for TL,TR,BR,BL
    if _polygon_signed_area(P_ccw) > 0:  # CCW -> flip to CW
        P_ccw = P_ccw[[0, 3, 2, 1]]

    return P_ccw  # TL, TR, BR, BL

def _bilinear_xy_in_quad(quad_xy_tl_tr_br_bl: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Bilinear interpolation inside quadrilateral:
      P00=TL, P10=TR, P11=BR, P01=BL.
      X(s,t) = (1-s)(1-t)P00 + s(1-t)P10 + s t P11 + (1-s)t P01
      s in [0,1] across columns, t in [0,1] across rows.
    """
    P00, P10, P11, P01 = quad_xy_tl_tr_br_bl

    s = np.linspace(0.0, 1.0, W, dtype=np.float64)  # columns
    t = np.linspace(0.0, 1.0, H, dtype=np.float64)  # rows
    S, T = np.meshgrid(s, t)  # (H,W)

    one_s, one_t = 1.0 - S, 1.0 - T
    term00 = (one_s * one_t)[..., None] * P00
    term10 = (S * one_t)[..., None] * P10
    term11 = (S * T)[..., None] * P11
    term01 = (one_s * T)[..., None] * P01
    XY = term00 + term10 + term11 + term01  # (H,W,2)
    return XY

def reshape_in_polygon(
    four_points: Iterable[Tuple[float, float]],
    pcd: o3d.geometry.PointCloud,
    H: int,
    W: int,
) -> o3d.geometry.PointCloud:
    """
    Warp the H×W grid of points from 'pcd' so that its XY lies bilinearly inside
    the given 4-point polygon; preserve Z point-wise.

    Parameters
    ----------
    four_points : iterable of 4 (x,y)
        Polygon corners in any order. Auto-ordered to TL,TR,BR,BL.
    pcd : o3d.geometry.PointCloud
        Input cloud with exactly H*W points in row-major order.
    H, W : int
        Grid height and width.

    Returns
    -------
    o3d.geometry.PointCloud
        Reshaped cloud.
    """
    xyz = _pcd2xyz_grid(pcd, H, W)    # (H,W,3)
    z_vals = xyz[..., 2].copy()

    quad_xy_in = np.asarray(list(four_points), dtype=np.float64).reshape(4, 2)
    quad_xy = _order_quad_tl_tr_br_bl(quad_xy_in)     # TL,TR,BR,BL

    XY = _bilinear_xy_in_quad(quad_xy, H, W)          # (H,W,2)
    warped = np.dstack([XY, z_vals])                  # (H,W,3)
    
    return _xyz_grid2pcd(warped)

class BGPatternFuser(O3DGUI):
    def __init__(self, config: BGPatternFuserConfig):
        self.config = config
        self.it = 0
        self.shape = None
        self.projected_pcd = None
        assert not self.config.in_camera, "When requesting fusion, the in_camera must be False (i.e. cam2nwu rotation required)"
        super().__init__(self.config.visMode)

    def set_internal_shape(self, shape):
        if self.shape is None:
            self.shape = shape[:2]
        else:
            assert self.shape == shape[:2], "Inconsistent image shape detected in fuser"

    def fuse(self, cimg, rd_pcd_rad, bg_pcd_rad, bg_pcd_can, pose, filename):
        H, W, _ = cimg.shape
        rd_pcm_nwu = pcd2pcm(rd_pcd_rad, H, W)
        self.set_internal_shape(rd_pcm_nwu.shape)
        gep_pcm_nwu_nonscaled = computeGeps(self.shape, 62, pose)
        gep_pcm_nwu = gep_pcm_nwu_nonscaled * calc_scale_factor(np.min(rd_pcm_nwu[:,:,2]), 
            scaling=Scaling.MIN_Z, bgz=None, pc_to_be_rescaled=gep_pcm_nwu_nonscaled)
        
        
        cg_bg_pcd_can, T = orient_point_cloud_cgplane_global(bg_pcd_can)
        bg_pcd_rad_trans = apply_transform_points(np.asarray(bg_pcd_rad.points), T)
        rd_pcd_rad_trans = apply_transform_points(np.asarray(rd_pcd_rad.points), T)
        rd_pcd_rad_trans_pcd = o3d.geometry.PointCloud()
        rd_pcd_rad_trans_pcd.points = o3d.utility.Vector3dVector(rd_pcd_rad_trans)
        rd_pcm_rad_trans = pcd2pcm(rd_pcd_rad_trans_pcd, H, W)
        bg_pcd_rad_trans_pcd = o3d.geometry.PointCloud()
        bg_pcd_rad_trans_pcd.points = o3d.utility.Vector3dVector(bg_pcd_rad_trans)
        fused_pcm_nwu = unfold_depth(rd_pcm_rad_trans, bg_pcd_rad_trans_pcd, gep_pcm_nwu, H, W)
        def resacle_and_repose(pcd):
            pcd *= 113.86/NULL_SCALE_MIN_Z
            pcd[:,:, 0] += 65
            pcd[:,:, 1] += 21
            return pcd
        gep_pcm_nwu_scaled = resacle_and_repose(gep_pcm_nwu)
        fused_pcm_nwu_scaled = resacle_and_repose(fused_pcm_nwu)
        fused_pcd_nwu = pcm2pcd(fused_pcm_nwu_scaled)
        save_pcm_as_pcd(gep_pcm_nwu, "gep_pcm_nwu.pcd", color = [255, 255, 0])
        save_pcm_as_pcd(gep_pcm_nwu_scaled, "gep_pcm_nwu_scaled.pcd", color = [255, 255, 0])
        aa = load_pcd("atp_prj_canonical.pcd")
        aa = pcd2pcm(aa, H, W)
        aaa = resacle_and_repose(aa)
        save_pcm_as_pcd(aaa, "aaa.pcd", color = [255, 255, 0])


        iterlist = []
        a = gep_pcm_nwu[0,0,0]
        b = gep_pcm_nwu[0,0,1]
        iterlist.append((a,b))
        a = gep_pcm_nwu[0,W-1,0]
        b = gep_pcm_nwu[0,W-1,1]
        iterlist.append((a,b))
        a = gep_pcm_nwu[H-1,W-1,0]
        b = gep_pcm_nwu[H-1,W-1,1]
        iterlist.append((a,b))
        a = gep_pcm_nwu[H-1,0,0]
        b = gep_pcm_nwu[H-1,0,1]
        iterlist.append((a,b))
        fused_pcm_nwu_polygon = reshape_in_polygon(iterlist, fused_pcd_nwu, H, W)
        save_pcd(np.asarray(fused_pcm_nwu_polygon.points),
                    np.asarray(np.zeros_like(fused_pcm_nwu_polygon.points)),
                    "fused_pcm_nwu_polygon.pcd")
        
        az = pcm2pcd(aaa)
        prjc = reshape_in_polygon(iterlist, az, H, W)
        save_pcd(np.asarray(prjc.points),
                    np.asarray(np.zeros_like(prjc.points)),
                    "prjc.pcd")
        
        save_pcm_as_pcd(fused_pcm_nwu_scaled, "fused_pcm_nwu_scaled.pcd", color = [255, 0, 0])

        _pcd = pcm2pcd(fused_pcm_nwu, cimg)
        # ======== Write to disk ========
        _filename_fuse = os.path.join(self.config.fusion_dir, f"{filename}.pcd")
        _filename_gep = os.path.join(self.config.gep_dir, f"{filename}.pcd")
        if self.config.do_save:
            save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath=_filename_fuse)
            _pcd_gep = pcm2pcd(gep_pcm_nwu, cimg)
            save_pcd(np.asarray(_pcd_gep.points), np.asarray(_pcd_gep.colors), filepath=_filename_gep)
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
            self.scene.scene.add_geometry(name, _pcd, self._mat_points(5.0))
            fit_camera(self.scene.scene, [_pcd])

        
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