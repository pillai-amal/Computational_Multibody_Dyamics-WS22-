# -------------------------------------
# Computational multibody dynamics
# Exercise 4.1 
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from matplotlib import pyplot as plt

from animate_pole_on_cart import animate_pole_on_cart


if __name__ == '__main__':
    
    # ------ system definition ------

    # model parameters
    m1 = 0.1                    # mass of cart
    m2 = 0.2                    # mass of pendulum
    L = 0.5                     # length of pendulum
    theta_S = m2 * L**2 / 12    # rotational inertia of pendulum
    g = 9.81                    # gravitational acceleration

    # external force excitation
    F = lambda t: 0.0 * np.sin(t)

    # initial condition
    t0 = 0
    q0 = np.array([-0.3, np.pi/2])
    u0 = np.array([0, 0]) 

    #####
    # (a)
    #####

    # mass matrix
    def M(q):
        # ------
        # complete the implementation of the mass matrix.
        # ------
        return

    # generalized force
    # ------
    # implement the generalized force here.
    # ------

    # ------ simulation using explicit Euler method ------

    #####
    # (b)
    #####

    # time step
    dt = 5e-4
    # final time
    t1 = 10
    
    # initialize numerical solution
    t = np.arange(t0, t1, dt) # vector of time instants t_k, i.e., t= [t_0, t_1, ...]
    nt = len(t)               # number of time instants
    nq = 2                    # number of generalized coordinates
    q = np.zeros((nt, nq))    # initialize matrix containing the solution q = [q_0, q_1, ...]^T

    # save initial condition
    # ------
    # save the initial conditions into q and u here.
    # ------

    #####
    # (d)
    #####

    # ------
    # implement the explicit Euler method here
    # Hint:
    # for each k do a step (qk, uk -> qk1, uk1) of the explicit Euler method
    # ------


    # ------ plots and animation ------
    
    #####
    # (e)
    #####

    # plot results

    # ------
    # plot the x and alpha here.
    # ------

    #####
    # (f)
    #####
    exit()
    animate_pole_on_cart(t, q)

    


    


    


