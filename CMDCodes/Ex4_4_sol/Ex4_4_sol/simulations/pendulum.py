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
from PyCMD.solver.TrapezoidalImplicit import TrapezoidalImplicit

if __name__ == '__main__':
    
    # ------ system definition ------

    # model parameters
    m = 1                        # mass 
    L = 1                        # length of pendulum

    # initial condition
    q0 = np.pi/2
    u0 = 0 

    # create pole on cart system
    pendulum = Pendulum(m, L, q0, u0)

    # ------ simulation ------

    # time step
    dt = 1e-3
    # final time
    t1 = 10

    euler_impl = EulerImplicit(pendulum, t1, dt)
    ti, qi, ui = euler_impl.integrate()

    trapez = TrapezoidalImplicit(pendulum, t1, dt)
    tt, qt, ut = trapez.integrate()

    # ------ plots and animation ------

    # energy
    E = lambda q, u: (m * L**2 / 6) * u**2 + m * 9.81 * L / 2 * (1 - np.cos(q))

    # plot results

    fig, ax = plt.subplots(2)

    ax[0].plot(ti, qi, label="Eul.")
    ax[0].plot(tt, qt, label="trap.")
    ax[0].legend()
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("alpha")
    

    ax[1].plot(ti, E(qi, ui), label="Eul.")
    ax[1].plot(tt, E(qt, ut), label="trap.")
    ax[1].legend()
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("E")



    plt.show()


    


    


    


