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

        # residual form of the scheme
        def R(xk1):
            R = np.zeros(nq + nu)
            qk1 = xk1[:nq]
            uk1 = xk1[nq:]
            R[:nq] = qk1 - q[k] - self.dt * self.system.q_dot(t[k+1], qk1, uk1) # q_dot(t, q, u)
            R[nq:] = self.system.M(t[k+1], qk1) @ (uk1 - u[k])\
                - self.dt * self.system.h(t[k+1], qk1, uk1) # M_inv * u_dot - h = 0
            return R

        # Implicit Euler scheme (qk, uk -> qk1, uk1)
        for k in tqdm(range(nt-1)):
            
            xk = np.append(q[k], u[k])

            # solve residual form for xk1. Use xk as initial guess.
            sol = root(R, xk)
            xk1 = sol.x
            converged = sol.success

            # if xk1 could not be computed, raise an error.
            if not converged:
                raise ValueError("Time step is not converged!")
                
            # update state
            q[k+1], u[k+1] = self.system.step_callback(t[k+1], xk1[:nq], xk1[nq:])

        return t, q, u
