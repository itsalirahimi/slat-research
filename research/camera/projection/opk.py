import os
import sys

import cv2
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../lib")
from utils.conversion import pcm2pcd
from ioHandle.IOHandler import save_pcd
from kinematics.pose import Point6, Pose
from projection.helper import project3D
from kinematics.transformations import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z

depth_src = "data/usegeo/data/metric_depth/gt/2021-04-23_13-17-12_S2223314_DxO.csv"
img_src = "data/usegeo/data/rgb/2021-04-23_13-17-12_S2223314_DxO.jpg"
depth = np.loadtxt(depth_src, delimiter=',', dtype=np.float32)
color_img = cv2.imread(img_src)
omega = -3.49E-05
phi   = 0.030542361
kappa = -1.885580658
p6 = Point6(x=0,y=0,z=100)
pose = Pose(p6=p6)

projected_pc, mc = project3D(depth, pose, 81.14, do_rotate=False)
        

_pcd = pcm2pcd(projected_pc, color_img)
save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath="p1.pcd")


R3   = rotation_matrix_x(-omega)
R2   = rotation_matrix_y(-phi)
R1   = rotation_matrix_z(-kappa)
R4 = [[1,0,0],[0,-1,0],[0,0,-1]]
R = R1 @ R2 @ R3 @ R4
projected_pc = projected_pc @ R.T
_pcd = pcm2pcd(projected_pc, color_img)
save_pcd(np.asarray(_pcd.points), np.asarray(_pcd.colors), filepath="p2.pcd")
