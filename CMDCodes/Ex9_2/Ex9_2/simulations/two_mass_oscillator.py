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

    # force_law = LinearSpring(c, l0)
    force_law = LinearSpringDamper(c, d, l0)
  
    # initial condition
    y10 = 1.1
    y20 = 1.9
    y10_dot = 0
    y20_dot = 0

    # create pole on cart system
    two_mass_oscillator = System()

    # add first mass via linear guidance
    lin_guidance1 = LinearGuidanceRel(q0=y10, u0=y10_dot)
    mass1 = RigidBodyRel(m1, np.eye(3), lin_guidance1, two_mass_oscillator.origin)
    two_mass_oscillator.add([lin_guidance1, mass1])

    # add second mass via linear guidance
    lin_guidance2 = LinearGuidanceRel(q0=y20, u0=y20_dot)
    mass2 = RigidBodyRel(m2, np.eye(3), lin_guidance2, two_mass_oscillator.origin)
    two_mass_oscillator.add([lin_guidance2, mass2])

    # add left spring
    spring1 = TwoPointInteraction(force_law, two_mass_oscillator.origin, mass1)
    two_mass_oscillator.add(spring1)

    # add right spring
    spring2 = TwoPointInteraction(force_law, mass1, mass2)
    two_mass_oscillator.add(spring2)

    # assemble the system
    two_mass_oscillator.assemble()

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


    


    


    


