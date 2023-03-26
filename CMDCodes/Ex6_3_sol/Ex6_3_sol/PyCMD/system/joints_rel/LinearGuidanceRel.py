# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

class LinearGuidanceRel:
    def __init__(self, q0=None, u0=None):
        """Linear guidance allowing only  motion on line starting in P1 having e_x^1 as direction"""

        # number of degrees of freedom
        self.nq = 1
        self.nu = 1

        # initial condition, set to zero if noting is passed to constructor.
        self.q0 = np.zeros(self.nq) if q0 is None else np.array([q0])
        self.u0 = np.zeros(self.nu) if u0 is None else np.array([u0])

        self.is_assembled = False

    def assembler_callback(self):
        self.is_assembled = True

    # kinematic equation (natural coordinates)
    def q_dot(self, t, q, u):
        return u

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    def B(self, t, q):
        return np.eye(1)

    # joint kinematics

    # transformation matrix
    def P1P2_T(self, t, q):
        return np.eye(3)

    # partial derivative of transformation matrix
    def P1P2_T_q(self, t, q):
        return np.zeros((3, 3, 1))

    # relative displacement
    def P1_r_P1P2(self, t, q):
        return np.array([q[0], 0, 0])

    def P1_r_P1P2_q(self, t, q):
        r_q = np.zeros((3, 1))
        r_q[0, 0] = 1
        return r_q

    # relative velocity
    def P1_r_dot_P1P2(self, t, q, u):
        return np.array([u[0], 0, 0])

    def P1_r_dot_P1P2_q(self, t, q, u):
        return np.zeros((3, self.nq))

    # (local) relative jacobian
    def P1_J_P1P2(self, t, q):
        J = np.zeros((3, 1))
        J[0, 0] = 1
        return J

    def P1_J_P1P2_q(self, t, q):
        return np.zeros((3, 1, 1))

    # kappa
    def P1_kappa_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_kappa_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    def P1_kappa_P1P2_u(self, t, q, u):
        return np.zeros((3, 1))

    # relative angular velocity
    def P1_omega_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_omega_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    # (local) jacobian of rotation
    def P1_J_R_P1P2(self, t, q):
        return np.zeros((3, 1))

    def P1_J_R_P1P2_q(self, t, q):
        return np.zeros((3, 1, 1))

    # kappa of rotation
    def P1_kappa_R_P1P2(self, t, q, u):
        return np.zeros(3)

    def P1_kappa_R_P1P2_q(self, t, q, u):
        return np.zeros((3, 1))

    def P1_kappa_R_P1P2_u(self, t, q, u):
        return np.zeros((3, 1))
