# -------------------------------------
# Computational multibody dynamics
# Exercise 4.2
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

class PendulumOnCart:
    def __init__(self, m1, m2, L, F, q0, u0, t0=0):
        # save model parameters as class properties
        self.m1 = m1
        self.m2 = m2
        self.L = L
        self.F = F
        self.theta_S = m2 * L**2 / 12
        self.grav = 9.81

        # save initial conditions as model properties
        self.t0 = t0
        self.q0 = q0
        self.u0 = u0

        # set numbers of generalized coordinates/velocities
        self.nq = 2
        self.nu = 2

    # mass matrix
    def M(self, t, q):
        x, alpha = q
        return np.array([[self.m1 + self.m2,                      self.m2 * self.L * np.cos(alpha) / 2],
                         [self.m2 * self.L * np.cos(alpha) / 2,   self.theta_S + self.m2 * self.L**2 / 4]])

    # total force
    def h(self, t, q, u):
        x, alpha = q
        x_dot, alpha_dot = u
        return np.array([self.m2 * self.L / 2 * np.sin(alpha) * alpha_dot**2 + self.F(t),
                         - self.m2 * self.L / 2 * self.grav * np.sin(alpha)])
