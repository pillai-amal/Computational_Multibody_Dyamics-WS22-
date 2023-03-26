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

        #####
        # (d)
        #####
        # save initial condition
        # use a for loop to implement the time steps of the sceme

        return t, q, u
