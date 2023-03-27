# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from PyCMD.math import A_IB_basic


class RevoluteJointRel:
    def __init__(self, q0=None, u0=None):
        """Revolute joint with rotation about e_z^P1"""
        self.nq = 1
        self.nu = 1
        self.q0 = np.zeros(self.nq) if q0 is None else np.array([q0])
        self.u0 = np.zeros(self.nu) if u0 is None else np.array([u0])

        self.angle0 = self.q0[0]

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # kinematic equation
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q):
        return np.eye(1)

    # other functions
    def A_P1P2(self, t, q):
        return A_IB_basic(q[0]).z()

    def A_P1P2_q(self, t, q):
        A_P1P2_q = np.zeros((3, 3, 1))
        A_P1P2_q[:, :, 0] = A_IB_basic(q[0]).dz()
        return A_P1P2_q

    def P1_r_P1P2(self, t, q):
        return np.zeros(3)

    def P1_r_P1P2_q(self, t, q):
        return np.zeros((3, self.nq))

    def P1_v_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_v_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    def P1_J_P1P2(self, t, q):
        return np.zeros((3, 1))

    def P1_J_P1P2_q(self, t, q):
        return np.zeros((3, 1, 1))

    def P1_kappa_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_kappa_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    def P1_kappa_P1P2_u(self, t, q, u):
        return np.zeros((3, 1))

    def P1_omega_P1P2(self, t, q, u):
        return np.array([0, 0, u[0]])

    def P1_omega_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    def P1_J_R_P1P2(self, t, q):
        return np.array([[0], [0], [1]])

    def P1_J_R_P1P2_q(self, t, q):
        return np.zeros((3, 1, 1))

    def P1_kappa_R_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_kappa_R_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    def P1_kappa_R_P1P2_u(self, t, q, u):
        return np.zeros((3, 1))

