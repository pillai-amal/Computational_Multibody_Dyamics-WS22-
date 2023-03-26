# -------------------------------------
# Computational multibody dynamics
# Exercise 4.2 
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

class EulerExplicit:
    def __init__(self, system, t1, dt):

        # save input as class parameters
        self.system = system
        self.t0 = system.t0
        self.t1 = t1
        self.dt = dt

    def integrate(self): # function that integrates the system
        
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

        # Explicit Euler scheme (qk, uk -> qk1, uk1)
        for k in range(nt-1):

            # update state
            q[k+1] = q[k] + self.dt * u[k] # q_dot = u
            u[k+1] = u[k] + self.dt * np.linalg.solve(self.system.M(t[k], q[k]), self.system.h(t[k], q[k], u[k])) # u_dot = M_inv * h

        return t, q, u
