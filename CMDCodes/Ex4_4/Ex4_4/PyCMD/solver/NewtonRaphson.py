# -------------------------------------
# Computational multibody dynamics
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Newton-Raphson method

import numpy as np

from numpy.linalg import solve


def NewtonRaphson(R, x0, dR, TOL=1e-8, MAXITER=100):
    
    # initialization
    converged = False # initially not converged
    iter = 0          # initialize iteration counter

    x = x0            # initial guess
    Rx = R(x)         # initial residual

    # while not converged, update x
    while not converged:
        
        #####
        # (b)
        # Hint: the argument dR is the jacobian of the residual R. dR = dR(x)
        #####


        # Newton update
        # compute residual with updated x

        iter +=1 # add one to iteration counter

        # if tolerance is met or too many iterations are made, exit while loop and set converged to True
        if np.max(np.abs(Rx)) < TOL or iter > MAXITER:
            converged = True
            break
    
    return converged, x


    