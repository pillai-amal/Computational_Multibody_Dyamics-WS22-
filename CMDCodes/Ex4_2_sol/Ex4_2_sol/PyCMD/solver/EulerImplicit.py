# -------------------------------------
# Computational multibody dynamics
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Implicit Euler Method

import numpy as np
from tqdm import tqdm

from scipy.optimize import root
from numpy.linalg import solve

class EulerImplicit:
    def __init__(self, system, t1, dt):

        self.system = system
        self.t0 = system.t0
        
        # integration time
        self.t1 = (
            t1 if t1 > self.t0 else ValueError("t1 must be larger than initial time t0.")
        )
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

        def residuum(xk1):
            R = np.zeros(nq + nu)
            qk1 = xk1[:nq]
            uk1 = xk1[nq:]
            R[:nq] = qk1 - q[k] - self.dt * uk1 # beta = 0
            R[nq:] = self.system.M(t[k+1], qk1) @ (uk1 - u[k])\
                - self.dt * self.system.h(t[k+1], qk1, uk1) # u_dot = M_inv * h
            return R

        # Implicit Euler scheme (qk, uk -> qk1, uk1)
        for k in tqdm(range(nt-1)):
            
            x_k = np.append(q[k], u[k])

            sol = root(residuum, x_k)
            xk1 = sol.x
            converged = sol.success

            if not converged:
                raise ValueError("Newton-Raphson is not converged!sf")
                
            # update state
            q[k+1] = xk1[:nq]
            u[k+1] = xk1[nq:]

        return t, q, u
