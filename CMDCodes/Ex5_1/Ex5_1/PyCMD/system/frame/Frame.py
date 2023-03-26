# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from PyCMD.utility.check_time_derivatives import check_time_derivatives
from PyCMD.math.algebra import skew2ax


class Frame:
    def __init__(
        self,
        r_OP=np.zeros(3),
        r_OP_t=None,
        r_OP_tt=None,
        A_IB=np.eye(3),
        A_IB_t=None,
        A_IB_tt=None,
    ):
        self.r_OP__, self.r_OP_t__, self.r_OP_tt__ = check_time_derivatives(
            r_OP, r_OP_t, r_OP_tt
        )
        self.A_IB__, self.A_IB_t__, self.A_IB_tt__ = check_time_derivatives(
            A_IB, A_IB_t, A_IB_tt
        )

        self.nq = 0
        self.nu = 0

        self.q0 = np.array([])
        self.u0 = np.array([])

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    #########################################
    # helper functions
    #########################################

    def q_dot(self, t, q, u):
        return np.array([])

    def A_IB(self, t, q=None):
        return self.A_IB__(t)

    def A_IB_q(self, t, q=None):
        return np.array([]).reshape((3, 3, 0))

    def r_OP(self, t, q=None, B_r_SP=np.zeros(3)):
        return self.r_OP__(t) + self.A_IB__(t) @ B_r_SP

    def v_P(self, t, q=None, u=None, B_r_SP=np.zeros(3)):
        return self.r_OP_t__(t) + self.A_IB_t__(t) @ B_r_SP

    def J_P(self, t, q=None, B_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def J_P_q(self, t, q, B_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0, 0))

    def kappa_P(self, t, q=None, u=None, B_r_SP=np.zeros(3)):
        return self.r_OP_tt__(t) + self.A_IB_tt__(t) @ B_r_SP
    
    def kappa_P_q(self, t, q=None, u=None, B_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    # new
    def kappa_P_u(self, t, q=None, u=None, B_r_SP=np.zeros(3)):
        return np.array([]).reshape((3, 0))

    def B_Omega(self, t, q=None, u=None):
        B_omega_IB = self.A_IB__(t).T @ self.A_IB_t__(t)
        return skew2ax(B_omega_IB)

    def B_Omega_q(self, t, q=None, u=None):
        return np.array([]).reshape((3, 0))

    def B_J_R(self, t, q):
        return np.array([]).reshape((3, 0))

    def B_J_R_q(self, t, q):
        return np.array([]).reshape((3, 0, 0))

    def B_kappa_R(self, t, q, u):
        B_kappa_IB = self.A_IB_t__(t).T @ self.A_IB_t__(t) + self.A_IB__(
            t
        ).T @ self.A_IB_tt__(t)
        return skew2ax(B_kappa_IB)

    def B_kappa_R_q(self, t, q, u):
        return np.array([]).reshape((3, 0))
    
    def B_kappa_R_u(self, t, q, u):
        return np.array([]).reshape((3, 0))
    