# -------------------------------------
# Computational multibody dynamics
#
# 23.01.23 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Inverse kinematics and inverse dynamics of the 2D robot arm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from PyCMD.math import IB_T_elem
from PyCMD.system import System, RigidBodyRel, RevoluteJointRelActuated, LinearGuidanceRelActuated
from PyCMD.solver import EulerExplicit, EulerImplicit, TrapezoidalImplicit

from tqdm import tqdm

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

    # remark: for the initialization of the forces, any function can be used since they will be overwritten by the values gotten from the inverse dynamics problem.
    F = lambda t: 0                     # force of linear motor (initialize with zero force)
    M = lambda t: 0                     # torque of rotational motor (initialize with zero moment)

    # initial generalized coordinates
    phi0 = np.pi / 4
    x0 = 2 * a1

    # planned trajectory
    B2_r_S2A = np.array([a2, 0, 0])
    r_OA0 = (x0 + a2) * np.array([np.cos(phi0), np.sin(phi0), 0])

    # line
    # d = np.array([0, -1, 0])  # direction
    # r_OA_g = lambda t: r_OA0 + t * d
    # r_OA_g_dot = lambda t: d
    # r_OA_g_ddot = lambda t: np.zeros(3)

    # # circle
    center = np.array([1.75, 1.75, 0])  # direction
    r_CA0 = r_OA0 - center
    omega = 3
    r_OA_g = lambda t: center + IB_T_elem(omega * t).z() @ r_CA0
    r_OA_g_dot = lambda t:  omega * IB_T_elem(omega * t).dz() @ r_CA0
    r_OA_g_ddot = lambda t: - omega**2 * IB_T_elem(omega * t).z() @ r_CA0

    # system definition
    system = System()

    revolute_joint = RevoluteJointRelActuated(tau=M, q0=phi0)
    r_OS10 = a1 * np.array([np.cos(phi0), np.sin(phi0), 0])
    RB1 = RigidBodyRel(m1, K_Theta_S1, revolute_joint, system.origin, B2_r_S2P2=np.array([-a1, 0, 0]))
    system.add((revolute_joint, RB1))

    lin_guidance = LinearGuidanceRelActuated(tau=F, q0=x0)
    r_OS10 = x0 * np.array([np.cos(phi0), np.sin(phi0), 0])
    RB2 = RigidBodyRel(m2, K_Theta_S2, lin_guidance, RB1, B1_r_S1P1=np.array([-a1, 0, 0]))
    system.add((lin_guidance, RB2))

    system.assemble()

    ############################################################################
    #                   inverse kinematics and inverse dynamics
    ############################################################################
    t0 = 0
    t1 = 2 * np.pi
    dt = 1e-2

    # initialize numerical solution
    t = np.arange(t0, t1, dt)
    nt = len(t)
    nq = system.nq
    nu = system.nu
    ntau = system.ntau
    q = np.zeros((nt, nq))
    u = np.zeros((nt, nu))
    u_dot = np.zeros((nt, nu))
    tau = np.zeros((nt, ntau))

    # save initial position
    q[0] = system.q0

    ##########
    # (c)
    # compute initial velocity
    J_A = RB2.J_P(t[0], q[0, RB2.qDOF], B_r_SP=B2_r_S2A)
    J_A_pinv = np.linalg.pinv(J_A)   # Moore-Penrose pseudo-inverse of J_A
    nu_A = RB2.nu_P(t[0], q[0, RB2.qDOF], B_r_SP=B2_r_S2A)
    u[0] = J_A_pinv @ (r_OA_g_dot(t[0]) - nu_A)
    system.u0 = u[0]
    ##########

    for k in tqdm(range(nt-1)):
        ##########
        # (d)
        # implement the inverse kinematics and inverse dynamics algorithm here.
        ##########
        

    ############################################################################
    #                   simulate system using planned forces
    ############################################################################

    # set all actuator forces to the computed values stemming from ID.
    system.set_tau(t, tau)

    # simulate the system using the computed forces from ID.
    dt_sim = 1e-2
    # solver = TrapezoidalImplicit(system, t1, dt_sim)
    solver = EulerExplicit(system, t1, dt_sim)

    t_sim, q_sim, u_sim = solver.integrate()

    ############################################################################
    #                   plot IK and simulated trajectory
    ############################################################################

    plt.plot(t, q[:,0], label='phi_IK')
    plt.plot(t_sim, q_sim[:,0], 'x',label='phi_sim')
    plt.plot(t, q[:,1], label='x_IK')
    plt.plot(t_sim, q_sim[:,1], 'x',label='x_sim')
    plt.legend()

    plt.show()


    ############################################################################
    #                   animate planned trajectory
    ############################################################################
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    scale = 2 * (x0 + a2)
    ax.axis('equal')
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    plt.title('IK trajectory')

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

    ############################################################################
    #                   animate simulated trajectory
    ############################################################################
    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    scale = 2 * (x0 + a2)
    ax.axis('equal')
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    plt.title('simulated trajectory')

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
        update(t_sim[i], q_sim[i], A, arm1, arm2, trace)

    # compute naimation interval according to te - ts = frames * interval / 1000
    frames = len(t_sim)
    interval = dt_sim * 1000
    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )
    plt.show()
