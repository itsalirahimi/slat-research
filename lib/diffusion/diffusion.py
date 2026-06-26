import cv2
import open3d as o3d
import numpy as np
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")
from diffusion.make_full_pcd import make_full_pcd
from fusion.helper import compute_center_of_mass
from geom.surfaces import cg_centeric_xy_spline, bspline_surface_mesh_from_ctrl, \
    project_external_along_normals_noreject
from projection.helper import project3D
from utils.conversion import pcd2hw1, pcm2pcd, pcd2pcdArr, pcdArr2pcd, pcm2pcdArr
# from utils.o3dviz import visualize_spline_mesh
from .tunning import Optimizer
from .scoring import Projection3DScorer
from .config import BGPatternDiffuserConfig
from .helper import RandomSurfacer, compute_shifted_ctrl_points, \
    create_grid_on_surface, mask2image
from utils.o3dviz import visualize_spline_mesh
import time
from kinematics.clouds import orient_point_cloud_cgplane_global, apply_transform_points
from ioHandle.IOHandler import save_pcd
from geom.surfaces import filter_mesh_neighbors

class BGPatternDiffuser:
    def __init__(self, config: BGPatternDiffuserConfig):
        self.config = config
        self.scorer = Projection3DScorer(self.config)
        self.tunner = Optimizer(self.config)
        self.scoring_base_mesh = None
    
    def initialCoarseTune(self, rd_pcd_arr, rd_pcd):
        print(f"[BGPatternDiffuser] ------- Initial coarse tunning ...")
        base_mesh, ctrl_pts, W, H, center_xy, z0 = cg_centeric_xy_spline(
            rd_pcd_arr, self.config.coarsetune_grid_w, self.config.coarsetune_grid_h, 
            self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v, 
            self.config.spline_mesh_marginal_ratio)
        
        if self.config.viz: # Pre-coarse-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ctrl_pts))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, base_mesh, rd_pcd, name="before initial coarse tune model")

        rs = RandomSurfacer(cloud=rd_pcd_arr, grid_w=self.config.coarsetune_grid_w, 
                            grid_h=self.config.coarsetune_grid_h, 
                            samples_u=self.config.spline_mesh_samples_u,
                            samples_v=self.config.spline_mesh_samples_v,
                            margin=self.config.spline_mesh_marginal_ratio)

        self.scorer.reset(rd_pcd_arr, smoothness_base_mesh=base_mesh, max_dz=2*rs.OF, fine_tune=True, 
                          sb_ds=False, original_colors=rd_pcd.colors)
        coarse_tunned_z1 = self.tunner.tune(ctrl_pts.copy(), self.scorer, 
                                                 iters=self.config.ct1iters,
                                                 alpha=self.config.tunning_alpha)
        ct1 = ctrl_pts.copy()
        ct1[:,2] = coarse_tunned_z1
        mbase = bspline_surface_mesh_from_ctrl(ct1, self.config.coarsetune_grid_w, self.config.coarsetune_grid_h,
                                       self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v)
        print(f"[BGPatternDiffuser] ------- Initial coarse tunning done")
        return mbase, ct1

    def diffuse(self, data, cimg, name, idx):
        rd_pcd_cam = data
        H, W, _ = cimg.shape
        print(np.asarray(data.points).shape)
        rd_pcd_cam = make_full_pcd(
            data,
            H=H,
            W=W,
            origin="upper",
            duplicate_strategy="max",
            interpolation_k=8,
        )
        print(np.asarray(rd_pcd_cam.points).shape)
        print(f" =============== [BGPatternDiffuser] Diffusing on index {idx}")
        t0 = time.time()
        rd_pcd, T = orient_point_cloud_cgplane_global(rd_pcd_cam)
        rd_pcd_arr = pcd2pcdArr(rd_pcd)

        self.scoring_base_mesh, self.guess1 = self.initialCoarseTune(rd_pcd_arr.copy(), rd_pcd)
        # cg_data = compute_center_of_mass(pcd2pcdArr(rd_pcd_cam))
        # # -------------- Provide initial guess if does not exist
        # if self.scoring_base_mesh is None:
        #     self.scoring_base_mesh, self.guess1 = self.initialCoarseTune(rd_pcd_arr.copy(), rd_pcd)
        # else:
        #     cg_old = cg_data
        #     cg_new = compute_center_of_mass(pcd2pcdArr(rd_pcd_cam))
        #     trans = cg_new - cg_old
        #     self.guess1 += trans

        # ================== Coarse-Tune the Back-Ground Model
        print(f"[BGPatternDiffuser] ------- Start Iterative coarse tunning on index {idx}")
        if self.config.viz: # Pre-coarse-tune viz
            c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.guess1))
            c.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(c, self.scoring_base_mesh, rd_pcd, 
                                  name="initial guess into coarse tune")

        _, _, sh, _ = compute_shifted_ctrl_points(rd_pcd_arr.copy(), self.guess1, 
                                        self.config.spline_mesh_samples_u,
                                        self.config.spline_mesh_samples_v, 1.0)

        self.scorer.reset(rd_pcd_arr.copy(), smoothness_base_mesh=self.scoring_base_mesh, 
                          max_dz=sh, original_colors=rd_pcd.colors)
        coarse_tunned_z = self.tunner.tune(self.guess1, self.scorer, 
                                           iters=self.config.coarsetune_iters,
                                           alpha=self.config.tunning_alpha)
        coarse_tunned = self.guess1.copy()
        coarse_tunned[:,2] = coarse_tunned_z

        self.guess1 = coarse_tunned.copy()
        self.scoring_base_mesh = bspline_surface_mesh_from_ctrl(self.guess1, 
                                                                self.config.coarsetune_grid_w, 
                                                                self.config.coarsetune_grid_h,
                                                                self.config.spline_mesh_samples_u, 
                                                                self.config.spline_mesh_samples_v)
        
        print(f"[BGPatternDiffuser] ------- Iterative Coarse tunning done on index {idx}")
        # ================= Prepare for Fine-Tunning
        _, nonshifted_mesh, sh, shifted_mesh = compute_shifted_ctrl_points(rd_pcd_arr.copy(), 
            coarse_tunned, self.config.spline_mesh_samples_u, self.config.spline_mesh_samples_v, 1.0)
        if self.config.viz: # Post-coarse-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coarse_tunned))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, nonshifted_mesh, 
                                  pcdArr2pcd(self.scorer.cloud_pts, self.scorer.original_colors), 
                                  name="post coarse tune model")

        upsampled_ctrl, _ = create_grid_on_surface(nonshifted_mesh, self.config.finetune_grid_w,
                                                   self.config.finetune_grid_h)

        if self.config.viz: # Pre-fine-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(upsampled_ctrl))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, nonshifted_mesh, rd_pcd, name="pre fine tune model")

        # # ================== Fine-Tune the Back-Ground Model
        print(f"[BGPatternDiffuser] ------- Start fine tunning on index {idx}")
        self.scorer.reset(rd_pcd_arr.copy(), smoothness_base_mesh=nonshifted_mesh, max_dz=sh, 
                          fine_tune=True, original_colors=rd_pcd.colors)
        fine_tunned_z = self.tunner.tune(upsampled_ctrl, self.scorer,
                                               iters=self.config.finetune_iters,
                                               alpha=self.config.tunning_alpha)
        fine_tunned = upsampled_ctrl.copy()
        fine_tunned[:,2] = fine_tunned_z

        fmesh = bspline_surface_mesh_from_ctrl(fine_tunned, self.config.finetune_grid_w, 
                                        self.config.finetune_grid_h, 
                                        self.config.spline_mesh_samples_u, 
                                        self.config.spline_mesh_samples_v)
        
        # _, _, _, fmesh = \
        #     compute_shifted_ctrl_points(rd_pcd_arr.copy(), fine_tunned, 
        #                                 self.config.spline_mesh_samples_u,
        #                                 self.config.spline_mesh_samples_v, 1.0)
        
        fil, bgmask = filter_mesh_neighbors(rd_pcd_arr, fine_tunned, self.config.spline_mesh_samples_u,
                                            self.config.spline_mesh_samples_v, 0.2, 1.2)
        fil_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fil))
        fil_pcd.paint_uniform_color([1.0, 0, 0])
        if self.config.viz: # Post-fine-tune viz
            ctrl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fine_tunned))
            ctrl_pcd.paint_uniform_color([1.0, 0.2, 0.2])
            visualize_spline_mesh(ctrl_pcd, fmesh, 
                                  pcdArr2pcd(self.scorer.cloud_pts, self.scorer.original_colors), 
                                  name="post fine tune model with scored point cloud",
                                  proj_pcd=fil_pcd)

        R = T[:3, :3]; t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3]  = -R.T @ t
        bgpts = project_external_along_normals_noreject(rd_pcd_arr, fmesh)
        # outname = "bgpts.pcd"
        # outname1 = "rd_pcd.pcd"
        # outname2 = "fine_tunned.pcd"
        # # Debug data log:
        # save_pcd(bgpts, np.zeros_like(np.asarray(rd_pcd.colors)), outname)
        # save_pcd(np.asarray(rd_pcd.points), np.asarray(rd_pcd_cam.colors), outname1)
        # np.savetxt(outname2, fine_tunned, delimiter=',')
        # Main data log:
        bg_pc_arr_cam = apply_transform_points(bgpts, T_inv)
        colors = np.zeros_like(np.asarray(rd_pcd_cam.colors))
        colors[:,0] = 1
        mask = mask2image(bgmask, H, W)
        _bg_pcd = pcdArr2pcd(bg_pc_arr_cam, colors)
        _bg_mhw1 = pcd2hw1(_bg_pcd, H, W)
        # ======== Write to disk ========
        _filename_diffusion = os.path.join(self.config.diffusion_dir, f"{name}.pcd")
        _filename_mask = os.path.join(self.config.mask_dir, f"{name}.png")
        _filename_mbg = os.path.join(self.config.mbg_dir, f"{name}.npy")
        if self.config.do_save:
            save_pcd(np.asarray(_bg_pcd.points), np.asarray(_bg_pcd.colors), filepath=_filename_diffusion)
            cv2.imwrite(_filename_mask, mask)
            np.save(_filename_mbg, _bg_mhw1)


            print(f"[BGPatternDiffuser] Fine tunning done on index {idx}, data written to {_filename_diffusion}. Time: {(time.time() - t0):.2f} sec")
