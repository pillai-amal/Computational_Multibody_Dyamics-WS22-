# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from PyCMD.math import ax2skew, ax2skew_a

# Euler parameters

def P1P2_T(p):
    a0 = p[0]
    a = p[1:]
    a_tilde = ax2skew(a)
    return np.eye(3) + 2 * (a0 * a_tilde + a_tilde @ a_tilde)

def P1P2_T_p(p):
    a0 = p[0]
    a = p[1:]
    a_tilde = ax2skew(a)
    a_tilde_a = ax2skew_a()
    P1P2_T_p = np.zeros((3, 3, 4))

    # partial derivative w.r.t. a0
    P1P2_T_p[:, :, 0] = 2 * a_tilde
    # partial derivative w.r.t. a
    P1P2_T_p[:, :, 1:] = 2 * (a0 * a_tilde_a + np.einsum('ij,jkl->ikl', a_tilde, a_tilde_a) + np.einsum('ijl,jk->ikl', a_tilde_a, a_tilde))
    return P1P2_T_p

def H(p):
    a0 = p[0]
    a = p[1:]
    H = np.zeros((4, 3))
    H[0] = - 0.5 * a
    H[1:] = 0.5 * (a0 * np.eye(3) + ax2skew(a))
    return H

def H_p(p):
    
    H_p = np.zeros((4, 3, 4))

    # partial derivative w.r.t. a0
    H_p[1:, :, 0] = 0.5 * np.eye(3)

    # partial derivative w.r.t. a
    H_p[0, :, 1:] = - 0.5 * np.eye(3)
    H_p[1:, :, 1:] = 0.5 * ax2skew_a()

    return H_p


class SphericalJointRel_EulerParameters:
    def __init__(self, q0=None, u0=None):
        """Spherical joint parametrized by Euler angles and angular velocities represented w.r.t. body fixed frame."""
        
        # degrees of freedom
        # q = p = (a0, a)
        # u = P1_omega_P1P2

        # number of degrees of freedom
        self.nq = 4 # number of Euler parameters
        self.nu = 3 # number of angular velocities

        # initial condition, set to zero if noting is passed to constructor.
        self.q0 = np.array([1, 0, 0, 0]) if q0 is None else np.array(q0)
        self.u0 = np.zeros(self.nu) if u0 is None else np.array(u0)

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # def step_callback(self, t, q, u):
    #     return q/np.linalg.norm(q), u

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

