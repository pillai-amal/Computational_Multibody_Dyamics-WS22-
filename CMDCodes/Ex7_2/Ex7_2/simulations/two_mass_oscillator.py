# -------------------------------------
# Computational multibody dynamics
#
# 20.12.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Simulation of two mass oscillator

import numpy as np
from matplotlib import pyplot as plt

from PyCMD.system import System, LinearGuidanceRel, RigidBodyRel, TwoPointInteraction
from PyCMD.system import LinearSpring, LinearSpringDamper
from PyCMD.solver import TrapezoidalImplicit


if __name__ == '__main__':
    
    # ------ system definition ------

    # model parameters
    m1 = 1          # mass 1
    m2 = 1          # mass 2
    c = 10          # spring stiffness
    l0 = 1          # undeformed lenght of spring
    d = 1           # damping ratio

    force_law = LinearSpring(c, l0)
  
    # initial condition
    y10 = 1.1
    y20 = 1.9
    y10_dot = 0
    y20_dot = 0

    # create pole on cart system
    two_mass_oscillator = System()

    #########
    # (b)
    #########

    # ------ simulation ------

    # initialize solver
    dt = 1e-2   # time step
    t1 = 5      # final time

    solver = TrapezoidalImplicit(two_mass_oscillator, t1, dt)
    t, q, u = solver.integrate()

    # ------ plots and animation ------

    # plot results

    plt.plot(t, q[:,0], label='position mass 1')
    plt.plot(t, q[:,1], label='position mass 2')
    plt.legend()

    plt.show()


    


    


    


