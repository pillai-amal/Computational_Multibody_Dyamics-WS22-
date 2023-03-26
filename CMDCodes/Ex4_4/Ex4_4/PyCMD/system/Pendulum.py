# -------------------------------------
# Computational multibody dynamics
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# planar physical pendulum

import numpy as np

class Pendulum:
    def __init__(self, m, L, q0, u0, t0=0):
        # model parameters

        self.m = m
        self.L = L
        self.theta_S = m * L**2 / 12
        self.grav = 9.81

        self.t0 = t0
        self.q0 = q0
        self.u0 = u0

        self.nq = 1
        self.nu = 1

        self.theta_P = self.theta_S + m * L**2 / 4

    # mass matrix
    def M(self, t, q):
        return np.array([[self.theta_P]])

    # mass matrix times u derived w.r.t. q
    def Mu_q(self, t, q, u):
        return np.zeros((self.nu, self.nq))

    # total force
    def h(self, t, q, u):
        return np.array([- self.m * self.grav * (self.L / 2) * np.sin(q)])

    # total force derived w.r.t. q
    def h_q(self, t, q, u):
        return np.zeros((self.nu, self.nq))

    # total force derived w.r.t. u
    def h_u(self, t, q, u):
        return np.array([[- self.m * self.grav * (self.L / 2) * np.cos(q)]])
