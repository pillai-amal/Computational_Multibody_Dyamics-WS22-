# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Simulation of pole on cart system

import numpy as np
from matplotlib import pyplot as plt

from animate_pole_on_cart import animate_pole_on_cart

from PyCMD.system import System, LinearGuidanceRel, RevoluteJointRel, RigidBodyRel, Force
from PyCMD.solver import TrapezoidalImplicit

if __name__ == '__main__':
    
    # ------ system definition ------

    # model parameters
    m1 = 0.1
    B1_theta_S1 = np.eye(3)
    m2 = 0.2
    L = 0.5
    B2_theta_S2 = np.diag([1, 1, m2 * L**2 / 12]) 
    

    # external force excitation
    F = lambda t: 0.0 * np.sin(t)

    # initial condition
    x0 = -0.3
    x_dot0 = 0
    alpha0 = np.pi/2
    alpha_dot0 = 0

    # create pole on cart system
    pole_on_cart_system = System()

    # add cart via linear guidance
    lin_guidance = LinearGuidanceRel(q0=x0, u0=x_dot0)
    cart = RigidBodyRel(m1, B1_theta_S1, lin_guidance, pole_on_cart_system.origin)
    pole_on_cart_system.add([lin_guidance, cart])

    # add pendulum via revolute joint
    rev_joint = RevoluteJointRel(q0=alpha0, u0=alpha_dot0)
    B2_r_S2P2 = np.array([0, L/2, 0])
    pendulum = RigidBodyRel(m2, B2_theta_S2, rev_joint, cart, B2_r_S2P2=B2_r_S2P2)
    pole_on_cart_system.add([rev_joint, pendulum])

    # add gravity to pendulum
    g = 9.81
    gravity = Force(np.array([0, -m2 * g, 0]), pendulum)
    pole_on_cart_system.add(gravity)

    # assemble the system
    pole_on_cart_system.assemble()

    # ------ simulation ------

    # initialize solver
    dt = 1e-2   # time step
    t1 = 5      # final time

    solver = TrapezoidalImplicit(pole_on_cart_system, t1, dt)
    t, q, u = solver.integrate()

    # ------ plots and animation ------

    # plot results

    plt.plot(t, q[:,0], label='position cart')
    # plt.plot(t, u[:,0], label='velocity cart')
    plt.plot(t, q[:,1], label='angle pendulum')
    # plt.plot(t, u[:,1], label='velocity pendulum')
    plt.legend()

    plt.show()
    
    animate_pole_on_cart(t, q)

    


    


    


