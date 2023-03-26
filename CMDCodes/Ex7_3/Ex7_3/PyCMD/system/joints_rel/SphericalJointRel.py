# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from PyCMD.math import IB_T_elem

# Tait-Bryan angles

def P1P2_T(p):
    alpha, beta, gamma = p
    return IB_T_elem(alpha).x() @ IB_T_elem(beta).y() @ IB_T_elem(gamma).z()

def P1P2_T_p(p):
    alpha, beta, gamma = p
    P1P2_T_p = np.zeros((3, 3, 3))

    # partial derivative w.r.t. alpha
    P1P2_T_p[:, :, 0] = IB_T_elem(alpha).dx() @ IB_T_elem(beta).y() @ IB_T_elem(gamma).z()
    # partial derivative w.r.t. beta
    P1P2_T_p[:, :, 1] = IB_T_elem(alpha).x() @ IB_T_elem(beta).dy() @ IB_T_elem(gamma).z()
    # partial derivative w.r.t. gamma
    P1P2_T_p[:, :, 2] = IB_T_elem(alpha).x() @ IB_T_elem(beta).y() @ IB_T_elem(gamma).dz()

    return P1P2_T_p

def H(p):
    alpha, beta, gamma = p
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    return np.array([[     cg,      -sg,  0],
                     [cb * sg,  cb * cg,  0],
                     [-sb * cg, sb * sg, cb]]) / cb

def H_p(p):
    alpha, beta, gamma = p
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    H_p = np.zeros((3, 3, 3))

    # partial derivative w.r.t. beta
    H_p[:, :, 1] = (sb/(cb**2)) * np.array([[     cg,      -sg,  0],
                                            [cb * sg,  cb * cg,  0],
                                            [-sb * cg, sb * sg, cb]]) \
                    + np.array([[       0,         0,   0],
                                [-sb * sg,  -sb * cg,   0],
                                [-cb * cg,   cb * sg, -sb]]) / cb
    # partial derivative w.r.t. gamma
    H_p[:, :, 2] = np.array([[    -sg,       -cg,  0],
                             [cb * cg,  -cb * sg,  0],
                             [sb * sg,   sb * cg,  0]]) / cb

    return H_p


class SphericalJointRel:
    def __init__(self, q0=None, u0=None):
        """Spherical joint parametrized by Tait-Bryan angles and angular velocities represented w.r.t. body fixed frame."""
        
        # degrees of freedom
        # q = p = (alpha, beta, gamma)
        # u = P1_omega_P1P2

        # number of degrees of freedom
        self.nq = 3 # number of Tait-Bryan angles
        self.nu = 3 # number of angular velocities

        # initial condition, set to zero if noting is passed to constructor.
        self.q0 = np.zeros(self.nq) if q0 is None else np.array(q0)
        self.u0 = np.zeros(self.nu) if u0 is None else np.array(u0)

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # kinematic equation (natural coordinates)
    def q_dot(self, t, q, u):
        return H(q) @ u

    def q_ddot(self, t, q, u, u_dot):
        return H(q) @ u_dot + (H_p(q) @ self.q_dot(t, q, u)) @ u

    def B(self, t, q):
        return H(q)

    # joint kinematics

    # transformation matrix
    def P1P2_T(self, t, q):
        return P1P2_T(q)

    def P1P2_T_q(self, t, q):
        return P1P2_T_p(q)

    # relative displacement
    def P1_r_P1P2(self, t, q):
        return np.zeros(3)

    def P1_r_P1P2_q(self, t, q):
        return np.zeros((3, self.nq))

    # relative velocity
    def P1_r_dot_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_r_dot_P1P2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    # (local) relative jacobian
    def P1_J_P1P2(self, t, q):
        return np.zeros((3, self.nu))

    def P1_J_P1P2_q(self, t, q):
        return np.zeros((3, self.nu, self.nq))

    # kappa
    def P1_kappa_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_kappa_P1P2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def P1_kappa_P1P2_u(self, t, q, u):
        return np.zeros((3, self.nu))

    # relative angular velocity
    def P1_omega_P1P2(self, t, q, u):
        return u

    def P1_omega_P1P2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    # (local) jacobian of rotation
    def P1_J_R_P1P2(self, t, q):
        return np.eye(3)

    def P1_J_R_P1P2_q(self, t, q):
        return np.zeros((3, self.nu, self.nq))

    # kappa of rotation
    def P1_kappa_R_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_kappa_R_P1P2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    def P1_kappa_R_P1P2_u(self, t, q, u):
        return np.zeros((3, self.nu))

