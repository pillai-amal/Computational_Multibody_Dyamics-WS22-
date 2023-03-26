# -------------------------------------
# Computational multibody dynamics
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Rotations

import numpy as np
from math import sin, cos


class IB_T_elem:
    """Elementary rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = sin(phi)
        self.cp = cos(phi)

    def x(self) -> np.ndarray:
        """Rotation around x-axis."""
        return np.array([[1,       0,        0],\
                         [0, self.cp, -self.sp],\
                         [0, self.sp,  self.cp]])

    def dx(self) -> np.ndarray:
        """Derivative of Rotation around x-axis."""
        return np.array([[0,        0,        0],\
                         [0, -self.sp, -self.cp],\
                         [0,  self.cp, -self.sp]])

    def y(self) -> np.ndarray:
        """Rotation around y-axis."""
        return np.array([[ self.cp, 0, self.sp],\
                         [       0, 1,       0],\
                         [-self.sp, 0, self.cp]])

    def dy(self) -> np.ndarray:
        """Derivative of Rotation around y-axis."""
        return np.array([[-self.sp, 0,  self.cp],\
                         [       0, 0,        0],\
                         [-self.cp, 0, -self.sp]])

    def z(self) -> np.ndarray:
        """Rotation around z-axis."""
        return np.array([[self.cp, -self.sp, 0],\
                         [self.sp,  self.cp, 0],\
                         [      0,        0, 1]])


    def dz(self) -> np.ndarray:
        """Derivative of Rotation around z-axis."""
        return np.array([[-self.sp,  -self.cp,  0],\
                         [  self.cp,  -self.sp, 0],\
                         [        0,         0, 0]])

