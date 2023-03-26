# -------------------------------------
# Computational multibody dynamics
# Exercise 4.2 
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from matplotlib import pyplot as plt

from PyCMD.system.PendulumOnCart import PendulumOnCart
from PyCMD.solver.EulerExplicit import EulerExplicit

from animate_pole_on_cart import animate_pole_on_cart

if __name__ == '__main__':
    
    # ------ system definition ------

    # model parameters
    m1 = 0.1                    # mass of cart
    m2 = 0.2                    # mass of pendulum
    L = 0.5                     # length of pendulum

    # external force excitation
    F = lambda t: 0.0 * np.sin(t)

    # initial condition
    q0 = np.array([-0.3, np.pi/2])
    u0 = np.array([0, 0]) 

    # create pendulum on cart system (this class has to be implemented in (b))
    pendulum_cart_system = PendulumOnCart(m1, m2, L, F, q0, u0)

    # ------ simulation ------

    # time step
    dt = 5e-4
    # final time
    t1 = 10

    # create a solver instance for the simulation of the system (this class has to be implemented in (c))
    euler_expl = EulerExplicit(pendulum_cart_system, t1, dt)

    # simulate the system using the explicit Euler method
    t, q, u = euler_expl.integrate()


    # ------ plots and animation ------

    # plot results

    fig, ax = plt.subplots(2)

    ax[0].plot(t, q[:,0])
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[1].plot(t, q[:,1])
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("alpha")
    plt.show()

    animate_pole_on_cart(t, q)


    


    


    


