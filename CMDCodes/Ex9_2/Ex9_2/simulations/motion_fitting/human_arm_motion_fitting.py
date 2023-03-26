# -------------------------------------
# Computational multibody dynamics
#
# 26.01.23 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Inverse kinematics and inverse dynamics of the human arm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from PyCMD.system import System, RigidBodyRel, RevoluteJointRelActuated, RevoluteJointRel, Force
from PyCMD.system import LinearMotor as Muscle

from scipy.interpolate import interp1d
from scipy.optimize import minimize


from tqdm import tqdm

if __name__ == "__main__":

    ############################################################################
    #                   load and interpolate measured marker data
    ############################################################################
    noisy = False

    if noisy:
        r_OPi_g = np.load('simulations/motion_fitting/r_OPi_noisy.npy')
    else:
        r_OPi_g = np.load('simulations/motion_fitting/r_OPi.npy')
    t_g = np.load('simulations/motion_fitting/t.npy')

    # after the subsequent interpolation, the marker positions can be called as functions of time, e.g., r_OP1_g(t).
    r_OP1_g = interp1d(t_g, r_OPi_g[0], axis=0, fill_value="extrapolate")
    r_OP2_g = interp1d(t_g, r_OPi_g[1], axis=0, fill_value="extrapolate")
    r_OP3_g = interp1d(t_g, r_OPi_g[2], axis=0, fill_value="extrapolate")
    r_OP4_g = interp1d(t_g, r_OPi_g[3], axis=0, fill_value="extrapolate")
    r_OP5_g = interp1d(t_g, r_OPi_g[4], axis=0, fill_value="extrapolate")

    ############################################################################
    #                   define system model
    ############################################################################
    # parameters
    L1 = 1                              # length of upper arm
    L2 = 1                              # length of lower arm

    m1 = 1                              # mass of first arm
    m2 = 1                              # mass of second arm

    theta1 = m1 * L1**2 / 12            # rotational inertia first arm
    K_Theta_S1 = theta1 * np.eye(3)     # rotational inertia tensor first arm (only the zz-component matters)
        
    theta2 = m2 * L2**2 / 12            # rotational inertia second arm
    K_Theta_S2 = theta2 * np.eye(3)     # rotational inertia tensor second arm (only the zz-component matters)

    # marker positions in respective body fixed frames
    B1_r_S1P1 = np.array([0 , L1 / 2, 0])
    B1_r_S1P2 = np.array([0 , 0, 0])
    B1_r_S1P3 = np.array([0 , -L1 / 2, 0])
    B2_r_S2P4 = np.array([0 , 0, 0])
    B2_r_S2P5 = np.array([0 , -L2 / 2, 0])

    # assemble system class
    system = System()

    shoulder_joint = RevoluteJointRelActuated()
    upper_arm = RigidBodyRel(m1, K_Theta_S1, shoulder_joint, system.origin, B2_r_S2P2=np.array([0 , L1 / 2, 0]))
    system.add((shoulder_joint, upper_arm))

    elbow = RevoluteJointRel()
    lower_arm = RigidBodyRel(m2, K_Theta_S2, elbow, upper_arm, B1_r_S1P1=np.array([0, -L1 / 2, 0]), B2_r_S2P2=np.array([0, L2 / 2, 0]))
    system.add((elbow, lower_arm))

    biceps = Muscle(upper_arm, lower_arm, B_r_SP1=np.array([0 , L1 / 2, 0]), B_r_SP2=np.array([0, L2 / 2 * 0.9, 0]))
    system.add(biceps)

    gravity1 = Force(np.array([0, - 9.81 * m1, 0]), upper_arm)
    gravity2 = Force(np.array([0, - 9.81 * m2, 0]), lower_arm)
    system.add((gravity1, gravity2))

    system.assemble()

    ############################################################################
    #                   inverse kinematics and inverse dynamics
    ############################################################################
    t0 = 0
    t1 = np.pi * 1
    dt = 5e-3

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

    # compute initial configuration (k=0)
    #########
    # (b)
    def f(q0): # define cost function 
        error_P1 = 0.5 * np.linalg.norm(r_OP1_g(t0) - upper_arm.r_OP(t0, q0[upper_arm.qDOF], B_r_SP=B1_r_S1P1))**2
        error_P3 = 0.5 * np.linalg.norm(r_OP3_g(t0) - upper_arm.r_OP(t0, q0[upper_arm.qDOF], B_r_SP=B1_r_S1P3))**2
        error_P2 = 0.5 * np.linalg.norm(r_OP2_g(t0) - upper_arm.r_OP(t0, q0[upper_arm.qDOF], B_r_SP=B1_r_S1P2))**2
        error_P4 = 0.5 * np.linalg.norm(r_OP4_g(t0) - lower_arm.r_OP(t0, q0[lower_arm.qDOF], B_r_SP=B2_r_S2P4))**2
        error_P5 = 0.5 * np.linalg.norm(r_OP5_g(t0) - lower_arm.r_OP(t0, q0[lower_arm.qDOF], B_r_SP=B2_r_S2P5))**2
        return error_P1 + error_P2 + error_P3 + error_P4 + error_P5
   
    sol = minimize(f, np.zeros(nq), tol=1e-8)  # solve minimization problem for q0
    q[0] = sol.x
    #########
    for k in tqdm(range(nt-1)):
        tk1 = t[k+1]
        # inverse kinematics

        #########
        # (c)
        # position level


        #########
        # (e)
        # velocity level 

        #########
        # (f)
        # acceleration level
       
        # save inverse kinematics
        q[k+1] = qk1
        u[k+1] = uk1
        u_dot[k+1] = u_dotk1

        # inverse dynamics
        


    ############################################################################
    #                   plot IK and ID results
    ############################################################################

    
    plt.plot(t, q[:, 0], label='alpha')
    plt.plot(t, q[:, 1], label='beta')
    plt.xlabel('t')
    plt.ylabel('angle')
    plt.title("Angles")
    plt.legend()

    plt.show()

    #########
    # (g)

    ############################################################################
    #                   animate identified trajectory
    ############################################################################

    fig, ax = plt.subplots()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    scale = 1.25 * (L1 + L2)
    ax.axis('equal')
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    plt.title('IK trajectory')

    b = L1 / 10

    def init(t, q):
        # x_0, y_0, z_0 = np.zeros(3)
        # x_S1, y_S1, z_S1 = RB1.r_OP(t, q[RB1.qDOF])
        x_P1, y_P1, z_P1 = upper_arm.r_OP(t, q[upper_arm.qDOF], B_r_SP=np.array([- b / 2, L1 / 2, 0]))
        # x_S2, y_S2, z_S2 = RB2.r_OP(t, q[RB2.qDOF])
        x_P2, y_P2, z_P2 = lower_arm.r_OP(t, q[lower_arm.qDOF], B_r_SP=np.array([- b / 2, L2 / 2, 0]))
        x_A, y_A, z_A = lower_arm.r_OP(t, q[lower_arm.qDOF], B_r_SP=np.array([0, -L2 / 2, 0]))


        # (COM,) = ax.plot([x_0, x_S1, x_S2], [y_0, y_S1, y_S2], "ok")
        (A,) = ax.plot([x_A], [y_A], "or")
        (trace,) = ax.plot([x_A], [y_A], "-r")

        arm1 = patches.Rectangle((x_P1, y_P1),
                                 L1,
                                 b,
                                 edgecolor='k',
                                 facecolor='w')
        arm2 = patches.Rectangle((x_P2, y_P2),
                                 L2,
                                 b,
                                 edgecolor='k',
                                 facecolor='w')
        
        ax.add_patch(arm2)
        ax.add_patch(arm1)
        return A, arm1, arm2, trace

    def update(t, q,  A, arm1, arm2, trace):
        alpha, beta = q
        x_P1, y_P1, z_P1 = upper_arm.r_OP(t, q[upper_arm.qDOF], B_r_SP=np.array([- b / 2, L1 / 2, 0]))
        x_P2, y_P2, z_P2 = lower_arm.r_OP(t, q[lower_arm.qDOF], B_r_SP=np.array([- b / 2, L2 / 2, 0]))
        x_A, y_A, z_A = lower_arm.r_OP(t, q[lower_arm.qDOF], B_r_SP=np.array([0, -L2 / 2, 0]))

        A.set_data([x_A], [y_A])

        tmp = trace.get_xydata()
        tmp = np.append(tmp, [[x_A, y_A]], 0)
        
        trace.set_data(tmp[:, 0], tmp[:, 1])

        arm1.set_angle(np.rad2deg(alpha) - 90)
        arm1.set_xy([x_P1, y_P1])
        arm2.set_angle(np.rad2deg(alpha) + np.rad2deg(beta) - 90)
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
