# -------------------------------------
# Computational multibody dynamics
# Exercise 4.2
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from matplotlib import pyplot as plt

from PyCMD.system.Pendulum import Pendulum
from PyCMD.solver.EulerImplicit import EulerImplicit

if __name__ == '__main__':
    
    # ------ system definition ------

    # model parameters
    m = 1                        # mass 
    L = 1                        # length of pendulum

    # initial condition
    q0 = np.pi/2
    u0 = 0 

    # create pendulum system
    pendulum = Pendulum(m, L, q0, u0)

    # ------ simulation ------

    # time step
    dt = 5e-4
    # final time
    t1 = 10

    euler = EulerImplicit(pendulum, t1, dt)

    t, q, u = euler.integrate()

    # ------ plots and animation ------

    # plot results

    fig, ax = plt.subplots(2)

    ax[0].plot(t, q[:])
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("alpha")
    ax[1].plot(t, u[:])
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("alpha_dot")

    plt.show()


    


    


    


