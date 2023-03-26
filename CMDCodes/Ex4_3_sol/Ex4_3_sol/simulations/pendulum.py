# -------------------------------------
# Computational multibody dynamics
# Exercise 4.3
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from matplotlib import pyplot as plt

from PyCMD.system.Pendulum import Pendulum
from PyCMD.solver.EulerExplicit import EulerExplicit
from PyCMD.solver.EulerImplicit import EulerImplicit
from PyCMD.solver.EulerSymplectic import EulerSymplectic

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

    # simulate the system using the implicit Euler scheme
    euler_impl = EulerImplicit(pendulum, t1, dt)
    ti, qi, ui = euler_impl.integrate()

    # simulate the system using the explicit Euler scheme (this is the compact version of the above)
    te, qe, ue = EulerExplicit(pendulum, t1, dt).integrate()

    # simulate the system using the symplectic Euler scheme
    euler_sympl = EulerSymplectic(pendulum, t1, dt)
    ts, qs, us = euler_sympl.integrate()

    # ------ plots and animation ------

    # energy as a function of q and u
    E = lambda q, u: (m * L**2 / 6) * u**2 + m * 9.81 * L / 2 * (1 - np.cos(q))

    # plot results

    fig, ax = plt.subplots(2)

    ax[0].plot(ti, qi, label="impl.")
    ax[0].plot(te, qe, label="expl.")
    ax[0].plot(ts, qs[:], label="sympl.")
    ax[0].legend()
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("alpha")
    

    ax[1].plot(ti, E(qi, ui), label="impl.")
    ax[1].plot(te, E(qe, ue), label="expl.")
    ax[1].plot(ts, E(qs, us), label="sympl.")
    ax[1].legend()
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E")

    plt.show()


    


    


    


