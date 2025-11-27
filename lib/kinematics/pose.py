from enum import Enum, auto
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from kinematics.transformations import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z

class Rot(Enum):
    RAD = auto()
    DEG = auto()

class RotFormat(Enum):
    OPK = auto()
    W2C_ROT = auto()
    AIRPLANE_EULER = auto()

def deg2rad(deg): return math.radians(deg)
def rad2deg(rad): return math.degrees(rad)

@dataclass
class OPK:
    x: float
    y: float
    z: float
    omega: float
    phi: float
    kappa: float
@dataclass
class Point6:
    x: float
    y: float
    z: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float

    def __init__(self, *, x, y, z, roll=0, pitch=0, yaw=0, rot: Rot = Rot.RAD):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if rot == Rot.RAD:
            self.roll_rad  = float(roll)
            self.pitch_rad = float(pitch)
            self.yaw_rad   = float(yaw)
        elif rot == Rot.DEG:
            self.roll_rad  = float(deg2rad(roll))
            self.pitch_rad = float(deg2rad(pitch))
            self.yaw_rad   = float(deg2rad(yaw))
        else:
            raise ValueError("rot must be Rot.RAD or Rot.DEG")

    @property
    def roll_deg(self):  return float(np.degrees(self.roll_rad))
    @property
    def pitch_deg(self): return float(np.degrees(self.pitch_rad))
    @property
    def yaw_deg(self):   return float(np.degrees(self.yaw_rad))


@dataclass
class ExtMat:
    data: NDArray[np.floating]


@dataclass
class Pose:
    p6: Point6
    extmat: NDArray[np.floating]  # 4x4

    def __init__(self, data, rot_format):
        if rot_format == RotFormat.AIRPLANE_EULER:
            self._init_from_p6(data)
        elif rot_format == RotFormat.OPK:
            self._init_from_opk(data)
        elif rot_format == RotFormat.W2C_ROT:
            R = data.data if isinstance(data, ExtMat) else data
            self._init_from_nwu2cam(np.asarray(R, dtype=float))

    def _init_from_p6(self, val: Point6) -> None:
        self.p6 = val
        Ry = np.array([[ 0, 0, 1], 
                       [0, 1, 0], 
                       [-1, 0, 0]])
        Rx = np.array([[1, 0, 0], 
                       [0, 0, -1], 
                       [0, 1, 0]])
        Rphi   = rotation_matrix_x(-val.roll_rad)
        Rtheta = rotation_matrix_y(-val.pitch_rad)
        Rpsi   = rotation_matrix_z(-val.yaw_rad)
        Rnwu   = rotation_matrix_x(np.pi)
        # rot = (Rnwu @ Rpsi @ Rtheta @ Rphi @ Rx @ Ry).astype(float)
        rot = (Rnwu @ Rpsi @ Rtheta @ Rphi).astype(float)
        extmat = np.eye(4, dtype=float)
        extmat[:3, :3] = rot
        extmat[:3,  3] = [val.x, val.y, val.z]
        self.extmat = extmat
    
    def _init_from_nwu2cam(self, extmat: ExtMat) -> None:
        rot = extmat[:3, :3]
        t  = extmat[:3, 3]
        self.p6 = Point6(x=extmat[0, 3], y=extmat[1, 3], z=extmat[2, 3])
        self.extmat = rot.astype(float)
    
    def _init_from_opk(self, opk: OPK) -> None:
        R3 = rotation_matrix_x(-opk.omega)
        R2 = rotation_matrix_y(-opk.phi)
        R1 = rotation_matrix_z(-opk.kappa)
        R4 = [[1,0,0],[0,-1,0],[0,0,-1]]
        R = R1 @ R2 @ R3 @ R4
        extmat = np.eye(4, dtype=float)
        extmat[:3, :3] = R
        extmat[:3,  3] = [opk.x, opk.y, opk.z]
        self.p6 = Point6(x=extmat[0, 3], y=extmat[1, 3], z=extmat[2, 3])
        self.extmat = extmat

    def getCAM2NWU(self):
        return (self.extmat[:3, :3]).T
    
    def getNWU2CAM(self):
        return self.extmat[:3, :3]
    