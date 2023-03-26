# -------------------------------------
# Computational multibody dynamics
#
# 13.01.23 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Simulation of the 2D robot arm

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from PyCMD.system import System, RigidBodyRel, RevoluteJointRelActuated, LinearGuidanceRelActuated
from PyCMD.solver import EulerExplicit, EulerImplicit

if __name__ == "__main__":

    # parameters
    m1 = 1                              # mass of first arm
    theta1 = 1                          # rotational inertia first arm
    K_Theta_S1 = theta1 * np.eye(3)     # rotational inertia tensor first arm (only the zz-component matters)
    
    m2 = 1                              # mass of second arm
    theta2 = 1                          # rotational inertia second arm
    K_Theta_S2 = theta2 * np.eye(3)     # rotational inertia tensor second arm (only the zz-component matters)
    
    a1 = 1                              # length of r_OS1
    a2 = 1                              # length of r_AS2

    F = lambda t: -1 * np.sin(3*t)                     # force of linear motor
    M = lambda t: t / (t + 1)                     # torque of rotational motor

    # initial conditions
    phi0 = 0
    phi_dot0 = 0

    x0 = 2 * a1
    x_dot0 = 0

    # system definition
    system = System()

    revolute_joint = RevoluteJointRelActuated(tau=M, q0=phi0, u0=phi_dot0)
    r_OS10 = a1 * np.array([np.cos(phi0), np.sin(phi0), 0])
    RB1 = RigidBodyRel(m1, K_Theta_S1, revolute_joint, system.origin, B2_r_S2P2=np.array([-a1, 0, 0]))
    system.add((revolute_joint, RB1))

    lin_guidance = LinearGuidanceRelActuated(tau=F, q0=x0, u0=x_dot0)
    r_OS10 = x0 * np.array([np.cos(phi0), np.sin(phi0), 0])
    RB2 = RigidBodyRel(m2, K_Theta_S2, lin_guidance, RB1, B1_r_S1P1=np.array([-a1, 0, 0]))
    system.add((lin_guidance, RB2))

    system.assemble()

    ############################################################################
    #                   solver
    ############################################################################
    t0 = 0
    t1 = 6
    dt = 1e-2

    solver = EulerExplicit(system, t1, dt)

    t, q, u = solver.integrate()

    ############################################################################
    #                   plot solution
    ############################################################################
    plt.plot(t, q[:,0], label='phi')
    plt.plot(t, q[:,1], label='x')
    plt.legend()

    plt.show()

    ############################################################################
    #                   animation
    ############################################################################
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    scale = 2 * (x0 + a2)
    ax.axis('equal')
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)

    def init(t, q):
        # x_0, y_0, z_0 = np.zeros(3)
        # x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_P1, y_P1, z_P1 = RB1.r_OP(t, q[RB1.qDOF], B_r_SP=np.array([-a1, -a1/4, 0]))
        # x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
        x_P2, y_P2, z_P2 = RB2.r_OP(t, q[RB2.qDOF], B_r_SP=np.array([-x0, -a1/8, 0]))
        x_A, y_A, z_A = RB2.r_OP(t, q[RB2.qDOF], B_r_SP=np.array([a2, 0, 0]))


        # (COM,) = ax.plot([x_0, x_S1, x_S2], [y_0, y_S1, y_S2], "ok")
        (A,) = ax.plot([x_A], [y_A], "or")
        (trace,) = ax.plot([x_A], [y_A], "-r")

        arm1 = patches.Rectangle((x_P1, y_P1),
                                 a1 * 3/2,
                                 a1/2,
                                 edgecolor='k',
                                 facecolor='w')
        arm2 = patches.Rectangle((x_P2, y_P2),
                                 x0 + a2,
                                 a1/4,
                                 edgecolor='k',
                                 facecolor='w')
        
        ax.add_patch(arm2)
        ax.add_patch(arm1)
        return A, arm1, arm2, trace

    def update(t, q,  A, arm1, arm2, trace):
        x = q[lin_guidance.qDOF][0]
        phi = q[revolute_joint.qDOF][0]
        # x_0, y_0, z_0 = np.zeros(3)
        # x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_P1, y_P1, z_P1 = RB1.r_OP(t, q[RB1.qDOF], B_r_SP=np.array([-a1, -a1/4, 0]))
        # x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
        x_P2, y_P2, z_P2 = RB2.r_OP(t, q[RB2.qDOF], B_r_SP=np.array([-x, -a1/8, 0]))
        x_A, y_A, z_A = RB2.r_OP(t, q[RB2.qDOF], B_r_SP=np.array([a2, 0, 0]))


        # COM.set_data([x_0, x_S1, x_S2], [y_0, y_S1, y_S2])
        A.set_data([x_A], [y_A])

        tmp = trace.get_xydata()
        tmp = np.append(tmp, [[x_A, y_A]], 0)
        
        trace.set_data(tmp[:, 0], tmp[:, 1])

        arm1.set_angle(np.rad2deg(phi))
        arm1.set_xy([x_P1, y_P1])
        arm2.set_width(x + a2)
        arm2.set_angle(np.rad2deg(phi))
        arm2.set_xy([x_P2, y_P2])


        return A, arm1, arm2, trace

    A, arm1, arm2, trace = init(0, q[0])

    def animate(i):
        update(t[i], q[i], A, arm1, arm2, trace)

    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t)
    interval = dt * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()
