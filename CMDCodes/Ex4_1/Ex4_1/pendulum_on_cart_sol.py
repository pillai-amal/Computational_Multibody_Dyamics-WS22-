# -------------------------------------
# Computational multibody dynamics
# Exercise 4.1 - solution
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
        x, alpha = q
        return np.array([[m1 + m2,                      m2 * L * np.cos(alpha) / 2],
                         [m2 * L * np.cos(alpha) / 2,   theta_S + m2 * L**2 / 4   ]])

    # generalized force
    def h(t, q, u):
        x, alpha = q
        x_dot, alpha_dot = u
        return np.array([m2 * L / 2 * np.sin(alpha) * alpha_dot**2 + F(t),
                         - m2 * L / 2 * g * np.sin(alpha)])


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
    u = np.zeros_like(q)      # initialize matrix containing the solution u = [u_0, u_1, ...]^T

    # save initial condition
    q[0] = q0
    u[0] = u0

    #####
    # (d)
    #####

    # Explicit Euler method (qk, uk -> qk1, uk1)
    for k in range(nt-1):

        # update state
        q[k+1] = q[k] + dt * u[k] # q_dot = u
        u[k+1] = u[k] + dt * np.linalg.solve(M(q[k]), h(t[k], q[k], u[k])) # u_dot = M_inv * h


    # ------ plots and animation ------
    
    #####
    # (e)
    #####

    # plot results

    fig, ax = plt.subplots(2)

    ax[0].plot(t, q[:,0])
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[1].plot(t, q[:,1])
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("alpha")

    plt.show()

    #####
    # (f)
    #####
    # exit()
    animate_pole_on_cart(t, q)

    


    


    


