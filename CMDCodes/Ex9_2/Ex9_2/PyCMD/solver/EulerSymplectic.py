# -------------------------------------
# Computational multibody dynamics
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Symplectic Euler Method

import numpy as np
from tqdm import tqdm

from scipy.optimize import root
from numpy.linalg import solve

class EulerSymplectic:
    def __init__(self, system, t1, dt):
        
        # save input as class properties
        self.system = system
        self.t0 = system.t0
        self.t1 = t1
        self.dt = dt

    def integrate(self):
        # initialize numerical solution
        t = np.arange(self.t0, self.t1, self.dt)
        nt = len(t)
        nq = self.system.nq
        nu = self.system.nu
        q = np.zeros((nt, nq))
        u = np.zeros((nt, nu))

        # save initial condition
        q[0] = self.system.q0
        u[0] = self.system.u0

        # Symplectic Euler scheme (qk, uk -> qk1, uk1)
        for k in tqdm(range(nt-1)):

            # update state
            uk1 = u[k] + self.dt * solve(self.system.M(t[k], q[k]), self.system.h(t[k], q[k], u[k])) # u_dot = M_inv * h
            qk1 = q[k] + self.dt * u[k+1] # q_dot = u

            q[k+1], u[k+1] = self.system.step_callback(t[k+1], qk1, uk1)
        return t, q, u
