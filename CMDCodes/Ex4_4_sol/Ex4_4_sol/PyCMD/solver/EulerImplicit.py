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

from PyCMD.solver.NewtonRaphson import NewtonRaphson

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

        # residual form of scheme
        def R(xk1):
            R = np.zeros(nq + nu)
            qk1 = xk1[:nq]
            uk1 = xk1[nq:]
            R[:nq] = qk1 - q[k] - self.dt * uk1 
            R[nq:] = self.system.M(t[k+1], qk1) @ (uk1 - u[k])\
                - self.dt * self.system.h(t[k+1], qk1, uk1) 
            return R

        # Jacobian of R (only needed for Newton-Raphson implementation in (c))
        def dR(xk1):
            dR = np.zeros((nq + nu, nq + nu))
            qk1 = xk1[:nq]
            uk1 = xk1[nq:]
            dR[:nq, :nq] = np.eye(nq) 
            dR[:nq, nq:] = - self.dt * np.eye(nu) 
            dR[nq:, :nq] = self.system.Mu_q(t[k+1], qk1, uk1 - u[k])\
                - self.dt * self.system.h_q(t[k+1], qk1, uk1)
            dR[nq:, nq:] = self.system.M(t[k+1], qk1) \
                - self.dt * self.system.h_u(t[k+1], qk1, uk1)
            return dR

        # Implicit Euler scheme (qk, uk -> qk1, uk1)
        for k in tqdm(range(nt-1)):
            
            xk = np.concatenate([q[k], u[k]])

            # solve residual for xk1 using the root method (replace this in (c))
            sol = root(R, xk)
            xk1 = sol.x
            converged = sol.success

            if not converged:
                raise ValueError("Time step is not converged!")
                
            # update state
            q[k+1] = xk1[:nq]
            u[k+1] = xk1[nq:]

        return t, q, u
