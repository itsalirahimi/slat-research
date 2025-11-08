import os
import sys



dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../lib")

import numpy as np
from fusion.fusion import reshape_in_polygon

from utils.conversion import pcd2pcm, pcm2pcd
from fusion.helper import unfold_depth
from kinematics.clouds import apply_transform_points, orient_point_cloud_cgplane_global

from ioHandle.IOHandler import load_pcd, save_pcd
H, W = 273, 365
bg_radial = load_pcd("data/ortholoc/diffusion/rad_test/background/L08_R0000.pcd")
bg_canonical = load_pcd("data/ortholoc/diffusion/can_test/background/L08_R0000.pcd")
prj_radial = load_pcd("data/ortholoc/projection/test/radial/L08_R0000.pcd")
prj_canonical = load_pcd("data/ortholoc/projection/test/canonical/L08_R0000.pcd")

cg_bg_canonical, T = orient_point_cloud_cgplane_global(bg_canonical)
atp_bg_radial = apply_transform_points(np.asarray(bg_radial.points), T)
atp_prj_radial = apply_transform_points(np.asarray(prj_radial.points), T)
atp_prj_canonical = apply_transform_points(np.asarray(prj_canonical.points), T)

save_pcd(np.asarray(cg_bg_canonical.points), np.asarray(np.zeros_like(cg_bg_canonical.points)), "cg_bg_canonical.pcd")
save_pcd(atp_bg_radial, np.asarray(np.zeros_like(cg_bg_canonical.points)), "atp_bg_radial.pcd")
save_pcd(atp_prj_radial, np.asarray(np.zeros_like(cg_bg_canonical.points)), "atp_prj_radial.pcd")


atp_bg_radial_pcd = load_pcd("atp_bg_radial.pcd")
atp_prj_radial_pcd = load_pcd("atp_prj_radial.pcd")
atp_prj_radial_pcm = pcd2pcm(atp_prj_radial_pcd, H, W)
unfolded = unfold_depth(atp_prj_radial_pcm, atp_bg_radial_pcd, atp_prj_radial_pcm, H, W)
save_pcd(atp_prj_canonical, np.asarray(np.zeros_like(cg_bg_canonical.points)), "atp_prj_canonical.pcd")

# unfolded_pcd = pcm2pcd(unfolded)
# save_pcd(np.asarray(unfolded_pcd.points), np.asarray(np.zeros_like(unfolded_pcd.points)), "unfolded.pcd")

# gep_pcd_nwu = load_pcd("data/ortholoc/fusion/aqq/ground/L08_R0000.pcd")
# gep_pcm_nwu = pcd2pcm(gep_pcd_nwu, H, W)

# iterlist = []
# a = gep_pcm_nwu[0,0,0]
# b = gep_pcm_nwu[0,0,1]
# iterlist.append((a,b))
# a = gep_pcm_nwu[0,W-1,0]
# b = gep_pcm_nwu[0,W-1,1]
# iterlist.append((a,b))
# a = gep_pcm_nwu[H-1,W-1,0]
# b = gep_pcm_nwu[H-1,W-1,1]
# iterlist.append((a,b))
# a = gep_pcm_nwu[H-1,0,0]
# b = gep_pcm_nwu[H-1,0,1]
# iterlist.append((a,b))
# unfolded_reshaped = reshape_in_polygon(iterlist, unfolded_pcd,H, W)

# atp_prj_canonical_pcd = pcm2pcd(atp_prj_canonical)
# atp_prj_canonical_reshaped = reshape_in_polygon(iterlist, atp_prj_canonical_pcd,H, W)

# save_pcd(np.asarray(unfolded_reshaped.points), np.asarray(np.zeros_like(unfolded_reshaped.points)), "unfolded_reshaped.pcd")
# save_pcd(np.asarray(atp_prj_canonical_reshaped.points), np.asarray(np.zeros_like(atp_prj_canonical_reshaped.points)), "atp_prj_canonical_reshaped.pcd")

