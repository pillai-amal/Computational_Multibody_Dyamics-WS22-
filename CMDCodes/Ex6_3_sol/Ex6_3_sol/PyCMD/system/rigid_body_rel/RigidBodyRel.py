# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

from PyCMD.math import cross3, ax2skew, ax2skew_a

class RigidBodyRel:
    def __init__(self,                  # this (successor) body (body 2 in lecture notes)
                m,                      # mass
                B_theta_S,              # rotational inertia tensor w.r.t. S in body fixed frame B
                joint,                  # joint used to attach this body to system
                predecessor_body,       # predecessor body (body 1 in lecture notes)
                B1_r_S1P1=np.zeros(3),  # position of P_1 w.r.t. S_1
                B1P1_T=np.eye(3),       # rel. rotation between B_1 and P_1
                B2_r_S2P2=np.zeros(3),  # position of P_2 w.r.t. S_2
                B2P2_T=np.eye(3)        # rel. rotation between B_2 and P_2
                ):

        # save arguments as class properties
        self.m = m
        self.B_theta_S = B_theta_S
        self.predecessor_body = predecessor_body
        self.joint = joint
        self.B1_r_S1P1 = B1_r_S1P1
        self.B1P1_T = B1P1_T
        self.B2_r_S2P2 = B2_r_S2P2
        self.P2B2_T = B2P2_T.T       # !!! we save the transposed of what we have in the lecture notes!!!

        self.is_assembled = False

    def assembler_callback(self):

        # this body can only be assembled if all preceeding bodies are assembled. Check that:
        if not self.joint.is_assembled:
            raise RuntimeError("Joint is not assembled; maybe not added to the model.")

        if not self.predecessor_body.is_assembled:
            raise RuntimeError(
                "Predecessor body is not assembled; maybe not added to the model."
            )

        # read out connectivity of this body and assemble initial condition
        self.qDOF = np.concatenate([self.predecessor_body.qDOF, self.joint.qDOF])
        self.q0 = np.concatenate([self.predecessor_body.q0, self.joint.q0])
        self.__nq = len(self.qDOF)
        self.nq1 = len(self.predecessor_body.qDOF)

        self.uDOF = np.concatenate([self.predecessor_body.uDOF, self.joint.uDOF])
        self.u0 = np.concatenate([self.predecessor_body.u0, self.joint.u0])
        self.__nu = len(self.uDOF)
        self.nu1 = len(self.predecessor_body.uDOF)

        # this body is now assembled
        self.is_assembled = True

    #####################
    # equations of motion
    #####################

    def M(self, t, q):
        J_S = self.J_P(t, q)
        B_J_R = self.B_J_R(t, q)
        return self.m * J_S.T @ J_S + B_J_R.T @ self.B_theta_S @ B_J_R
        
    def Mu_q(self, t, q, u):
        J_S = self.J_P(t, q)
        B_J_R = self.B_J_R(t, q)
        J_S_q = self.J_P_q(t, q)
        B_J_R_q = self.B_J_R_q(t, q)

        Mu_q = (
            np.einsum("ijl,ik,k->jl", J_S_q, J_S, self.m * u)
            + np.einsum("ij,ikl,k->jl", J_S, J_S_q, self.m * u)
            + np.einsum("ijl,ik,k->jl", B_J_R_q, self.B_theta_S @ B_J_R, u)
            + np.einsum("ij,jkl,k->il", B_J_R.T @ self.B_theta_S, B_J_R_q, u)
        )

        return Mu_q

    def f_gyr(self, t, q, u):
        B_Omega = self.B_Omega(t, q, u)
        return self.m * self.J_P(t, q).T @ self.kappa_P(t, q, u) \
                + self.B_J_R(t, q).T @ (self.B_theta_S @ self.B_kappa_R(t, q, u)\
                                        + cross3(B_Omega, self.B_theta_S @ B_Omega))

    def f_gyr_q(self, t, q, u):
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_q = self.B_Omega_q(t, q, u)
        tmp1 = self.B_theta_S @ self.B_kappa_R(t, q, u)
        tmp1_q = self.B_theta_S @ self.B_kappa_R_q(t, q, u)
        tmp2 = cross3(B_Omega, self.B_theta_S @ B_Omega)
        tmp2_q = (ax2skew(B_Omega) @ self.B_theta_S - ax2skew(self.B_theta_S @ B_Omega)) @ B_Omega_q

        f_gyr_q = (
            np.einsum("jik,j->ik", self.J_P_q(t, q), self.m * self.kappa_P(t, q, u))
            + self.m * self.J_P(t, q).T @ self.kappa_P_q(t, q, u)
            + np.einsum("jik,j->ik", self.J_P_q(t, q), tmp1 + tmp2)
            + self.B_J_R(t, q).T @ (tmp1_q + tmp2_q)
        )

        return f_gyr_q

    def f_gyr_u(self, t, q, u):
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_u = self.B_J_R(t, q)
        tmp1_u = self.B_theta_S @ self.B_kappa_R_u(t, q, u)
        tmp2_u = (ax2skew(B_Omega) @ self.B_theta_S - ax2skew(self.B_theta_S @ B_Omega)) @ B_Omega_u

        f_gyr_u = self.m * self.J_P(t, q).T @ self.kappa_P_u(t, q, u) + self.B_J_R(t, q).T @ (tmp1_u + tmp2_u)
        
        return f_gyr_u

    
    #########################################
    # rigid body kinematics - rotational
    #########################################

    # orientation of the body (transformation matrix from inertial to body fixed frame)
    def IB_T(self, t, q):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        return self.predecessor_body.IB_T(t, q1) @ self.B1P1_T @ self.joint.P1P2_T(t, qj) @ self.P2B2_T

    def IB_T_q(self, t, q):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        IB_T_q = np.zeros((3, 3, self.__nq))
        IB_T_q[:, :, : self.nq1] = np.einsum(
            "ijk,jl->ilk", self.predecessor_body.IB_T_q(t, q1),
                                    self.B1P1_T\
                                        @ self.joint.P1P2_T(t, qj)\
                                            @ self.P2B2_T
        )
        IB_T_q[:, :, self.nq1 :] = np.einsum(
            "ij,jkl,km->iml", self.predecessor_body.IB_T(t, q1)\
                                    @ self.B1P1_T,
                                        self.joint.P1P2_T_q(t, qj),
                                            self.P2B2_T
        )

        return IB_T_q

    # angular velocity of the body 
    def B_Omega(self, t, q, u):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]
        P1_Omega = self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1, u1) + self.joint.P1_omega_P1P2(t, qj, uj)
        return ( self.joint.P1P2_T(t, qj) @ self.P2B2_T ).T @ P1_Omega

    def B_Omega_q(self, t, q, u):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]
        P1_Omega_q = np.hstack([np.einsum("ji,jl->il",
                               self.B1P1_T,
                               self.predecessor_body.B_Omega_q(t, q1, u1)),
         self.joint.P1_omega_P1P2_q(t, qj, uj)])

        P1_Omega = self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1, u1) + self.joint.P1_omega_P1P2(t, qj, uj)
        B2P1_T_qj = np.einsum("ijl,jk->kil", self.joint.P1P2_T_q(t, qj), self.P2B2_T)
        B_Omega_q = (self.joint.P1P2_T(t, qj) @ self.P2B2_T).T @ P1_Omega_q

        B_Omega_q[:, self.nq1:] += np.einsum("ijk,j->ik", B2P1_T_qj, P1_Omega)
        
        return B_Omega_q

    # jacobian of rotation
    def B_J_R(self, t, q):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        J_R = np.zeros((3, self.__nu))
        B2P1_T = ( self.joint.P1P2_T(t, qj) @ self.P2B2_T ).T
        J_R[:, : self.nu1] = B2P1_T @ self.B1P1_T.T @ self.predecessor_body.B_J_R(t, q1)
        J_R[:, self.nu1 :] = B2P1_T @ self.joint.P1_J_R_P1P2(t, qj)

        return J_R

    def B_J_R_q(self, t, q):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        B_J_R_q = np.zeros((3, self.__nu, self.__nq))

        B2P1_T = ( self.joint.P1P2_T(t, qj) @ self.P2B2_T ).T
        # transpose via einsum li instead of il.T
        B2P1_T_q = np.einsum("ijk,jl->lik",
                             self.joint.P1P2_T_q(t, qj), self.P2B2_T)

        B_J_R_q[:, : self.nu1, : self.nq1] = np.einsum(
            "ij,jkl->ikl", B2P1_T @ self.B1P1_T.T,
            self.predecessor_body.B_J_R_q(t, q1)
        )
        B_J_R_q[:, : self.nu1, self.nq1 :] = np.einsum(
            "ijk,jl->ilk", B2P1_T_q,
            self.B1P1_T.T @ self.predecessor_body.B_J_R(t, q1)
        )
        B_J_R_q[:, self.nu1 :, self.nq1 :] = np.einsum(
            "ijk,jl->ilk", B2P1_T_q, self.joint.P1_J_R_P1P2(t, qj)
        ) + np.einsum("ij,jkl->ikl", B2P1_T, self.joint.P1_J_R_P1P2_q(t, qj)
        )

        return B_J_R_q

    # kappa of rotation
    def B_kappa_R(self, t, q, u):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]

        B2P1_T = ( self.joint.P1P2_T(t, qj) @ self.P2B2_T ).T
        P1_kappa_R = self.B1P1_T.T @ self.predecessor_body.B_kappa_R(t, q1, u1) + self.joint.P1_kappa_R_P1P2(t, qj, uj) + cross3(self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1, u1), self.joint.P1_omega_P1P2(t, qj, uj))
        return B2P1_T @ P1_kappa_R

    def B_kappa_R_q(self, t, q, u):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]

        B_kappa_R_q = np.zeros((3, self.__nq))
        
        B2P1_T = ( self.joint.P1P2_T(t, qj) @ self.P2B2_T ).T
        pred_B_Omega = self.predecessor_body.B_Omega(t, q1, u1)

        P1_kappa_R = self.B1P1_T.T @ self.predecessor_body.B_kappa_R(t, q1, u1)\
                + self.joint.P1_kappa_R_P1P2(t, qj, uj)\
                + cross3(self.B1P1_T.T @ pred_B_Omega , self.joint.P1_omega_P1P2(t, qj, uj))
            
        B_kappa_R_q[:, : self.nq1] = B2P1_T\
            @ (self.B1P1_T.T @ self.predecessor_body.B_kappa_R_q(t, q1, u1) \
                - ax2skew(self.joint.P1_omega_P1P2(t, qj, uj)) @  self.B1P1_T.T @ self.predecessor_body.B_Omega_q(t, q1, u1))
        B_kappa_R_q[:, self.nq1 :] = B2P1_T\
            @ (self.joint.P1_kappa_R_P1P2_q(t, qj, uj) + ax2skew(self.B1P1_T.T @ pred_B_Omega) @ self.joint.P1_omega_P1P2_q(t, qj, uj))\
            + np.einsum("ji,kjl,k->il", self.P2B2_T, self.joint.P1P2_T_q(t, qj), P1_kappa_R)
            

        return B_kappa_R_q

    def B_kappa_R_u(self, t, q, u):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]

        B2P1_T = ( self.joint.P1P2_T(t, qj) @ self.P2B2_T ).T
        B_kappa_R_u = np.zeros((3, self.__nu))

        B_kappa_R_u[:, : self.nu1] = B2P1_T\
            @ (self.B1P1_T.T @ self.predecessor_body.B_kappa_R_u(t, q1, u1)\
                + ax2skew(self.joint.P1_omega_P1P2(t, qj, uj)) @ self.B1P1_T.T @ self.predecessor_body.B_J_R(t, q1) )
        B_kappa_R_u[:, self.nu1 :] = B2P1_T\
            @ (self.joint.P1_kappa_R_P1P2_u(t, qj, uj) + ax2skew(self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1, u1)) @ self.joint.P1_J_R_P1P2(t, qj) )
        return B_kappa_R_u

    #########################################
    # rigid body kinematics - translational
    #########################################

    # position vector of point P addressed by B_r_SP w.r.t. center of mass S
    def r_OP(self, t, q, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        return self.predecessor_body.r_OP(t, q1, B_r_SP=self.B1_r_S1P1)\
            + self.predecessor_body.IB_T(t, q1) @ self.B1P1_T @ self.joint.P1_r_P1P2(t, qj)\
            + self.IB_T(t, q) @ (B_r_SP - self.B2_r_S2P2)
        
    # velocity of point P addressed by B_r_SP w.r.t. center of mass S
    def v_P(self, t, q, u, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]
        v_P2 = self.predecessor_body.v_P(t, q1, u1, B_r_SP=self.B1_r_S1P1) \
                + self.predecessor_body.IB_T(t, q1) @ self.B1P1_T @ \
                    (self.joint.P1_r_dot_P1P2(t, qj, uj)\
                + cross3(self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1, u1), self.joint.P1_r_P1P2(t, qj)))
        v_P2P = self.IB_T(t, q) @ cross3(self.B_Omega(t, q, u), B_r_SP - self.B2_r_S2P2)
        return v_P2 + v_P2P

    # jacobian of point P addressed by B_r_SP w.r.t. center of mass S
    def J_P(self, t, q, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
           
        IB1_T = self.predecessor_body.IB_T(t, q1)
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        
        J_P = - self.IB_T(t, q) @ ax2skew(B_r_P2P) @ self.B_J_R(t, q)

        J_P[:, :self.nu1] += self.predecessor_body.J_P(t, q1, B_r_SP=self.B1_r_S1P1) \
            - IB1_T @ self.B1P1_T @ ax2skew(self.joint.P1_r_P1P2(t, qj)) @ self.B1P1_T.T @ self.predecessor_body.B_J_R(t, q1)

        J_P[:, self.nu1:] += IB1_T @ self.B1P1_T @ self.joint.P1_J_P1P2(t, qj)
        return J_P

    def J_P_q(self, t, q, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]

        B_r_P2P = B_r_SP - self.B2_r_S2P2
        IB1_T = self.predecessor_body.IB_T(t, q1)
        IB1_T_q = self.predecessor_body.IB_T_q(t, q1)
        B1_J_R1 = self.predecessor_body.B_J_R(t, q1)
        
        J_P_q = np.einsum(
            "ij,jkl->ikl",
            - self.IB_T(t, q) @ ax2skew(B_r_P2P), self.B_J_R_q(t, q)
        )
        J_P_q -= np.einsum(
            "ijk,jl->ilk",
            self.IB_T_q(t, q), ax2skew(B_r_P2P) @ self.B_J_R(t, q)
        )
        J_P_q[:,
              : self.nu1,
              : self.nq1] += self.predecessor_body.J_P_q(t,
                                                         q1,
                                                         B_r_SP=self.B1_r_S1P1)
        J_P_q[:,
              : self.nu1,
              : self.nq1] -= np.einsum(
                "ij,jlk->ilk",
              IB1_T @ self.B1P1_T @ ax2skew(self.joint.P1_r_P1P2(t, qj)) @ self.B1P1_T.T , self.predecessor_body.B_J_R_q(t, q1))

        J_P_q[:,
              : self.nu1,
              : self.nq1] -= np.einsum(
                "ijk,jl->ilk",
              IB1_T_q, self.B1P1_T @ ax2skew(self.joint.P1_r_P1P2(t, qj)) @ self.B1P1_T.T @ B1_J_R1)

        J_P_q[:,
              : self.nu1,
               self.nq1 :] -= np.einsum(
                "ijk,jl->ilk",
              IB1_T @ self.B1P1_T @ ax2skew_a() @ self.joint.P1_r_P1P2_q(t, qj), self.B1P1_T.T @ B1_J_R1)

        J_P_q[:,
              self.nu1 :,
              : self.nq1] += np.einsum(
            "ijk,jl->ilk", IB1_T_q,
            self.B1P1_T @ self.joint.P1_J_P1P2(t, qj)
        )
        J_P_q[:,
              self.nu1 :,
              self.nq1 :] += np.einsum(
            "ij,jlk->ilk",
            IB1_T @ self.B1P1_T,
            self.joint.P1_J_P1P2_q(t, qj)
        )

        return J_P_q

    # kappa of point P addressed by B_r_SP w.r.t. center of mass S
    def kappa_P(self, t, q, u, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]

        B_r_P2P = B_r_SP - self.B2_r_S2P2

        P1_omega_IP1 = self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1,u1)

        kappa_P = self.predecessor_body.kappa_P(t, q1, u1, B_r_SP=self.B1_r_S1P1) \
            + self.predecessor_body.IB_T(t, q1) @ self.B1P1_T\
                    @ (self.joint.P1_kappa_P1P2(t, qj, uj) \
                        - cross3(self.joint.P1_r_P1P2(t, qj), self.B1P1_T.T @ self.predecessor_body.B_kappa_R(t, q1, u1))\
                        + 2 * cross3(P1_omega_IP1, self.joint.P1_r_dot_P1P2(t, qj, uj))\
                    + cross3(P1_omega_IP1, cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, qj))))

        kappa_P += self.IB_T(t, q) @ (
            cross3(self.B_kappa_R(t, q, u), B_r_P2P)
            + cross3(self.B_Omega(t, q, u), cross3(self.B_Omega(t, q, u), B_r_P2P))
        )

        return kappa_P


    def kappa_P_q(self, t, q, u, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]
        
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_q = self.B_Omega_q(t, q, u)
        P1_omega_IP1 = self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1,u1)
        P1_omega_IP1_q1 = self.B1P1_T.T @ self.predecessor_body.B_Omega_q(t, q1, u1)
        pred_kappa_P_R = self.predecessor_body.B_kappa_R(t, q1, u1)
        pred_IB_T = self.predecessor_body.IB_T(t, q1)
   

        tmp1 = cross3(self.B_kappa_R(t, q, u), B_r_P2P)
        tmp1_q = -ax2skew(B_r_P2P) @ self.B_kappa_R_q(t, q, u)
        tmp2 = cross3(B_Omega, cross3(B_Omega, B_r_P2P))
        tmp2_q = (
            -(ax2skew(cross3(B_Omega, B_r_P2P)) + ax2skew(B_Omega) @ ax2skew(B_r_P2P))
            @ B_Omega_q)

        kappa_P_q = np.einsum("ijk,j->ik", self.IB_T_q(t, q), tmp1 + tmp2)
        + self.IB_T(t, q) @ (tmp1_q + tmp2_q)
        

        kappa_P_q[:, : self.nq1] += self.predecessor_body.kappa_P_q(t, q1, u1)

        tmp3 = self.joint.P1_kappa_P1P2(t, qj, uj) \
                    + cross3(self.B1P1_T.T @ pred_kappa_P_R, self.joint.P1_r_P1P2(t, qj))\
                    + 2 * cross3(P1_omega_IP1, self.joint.P1_r_dot_P1P2(t, qj, uj))\
                    + cross3(P1_omega_IP1, cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, qj)))

        tmp3_q1 = - ax2skew(self.joint.P1_r_P1P2(t, qj)) @ self.B1P1_T.T @ self.predecessor_body.B_kappa_R_q(t, q1, u1)\
                  - 2 * ax2skew(self.joint.P1_r_dot_P1P2(t, qj, uj)) @ P1_omega_IP1_q1\
                  - (ax2skew(cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, qj))) + ax2skew(P1_omega_IP1) @ ax2skew(self.joint.P1_r_P1P2(t, qj))) @ P1_omega_IP1_q1
                  
        tmp3_qj = self.joint.P1_kappa_P1P2_q(t, qj, uj) \
                    + ax2skew(self.B1P1_T.T @ pred_kappa_P_R) @ self.joint.P1_r_P1P2_q(t, qj)\
                    + 2 * ax2skew(P1_omega_IP1) @ self.joint.P1_r_dot_P1P2_q(t, qj, uj)\
                    + ax2skew(P1_omega_IP1) @ ax2skew(P1_omega_IP1) @ self.joint.P1_r_P1P2_q(t, qj)

        kappa_P_q[:, : self.nq1] +=  np.einsum("ijk,j->ik", self.predecessor_body.IB_T_q(t, q1),
                        self.B1P1_T @ tmp3) + pred_IB_T@ self.B1P1_T @ tmp3_q1
        
        kappa_P_q[:, self.nq1 :] += pred_IB_T @ self.B1P1_T @ tmp3_qj
        return kappa_P_q

    def kappa_P_u(self, t, q, u, B_r_SP=np.zeros(3)):
        q1 = q[:self.nq1]
        qj = q[self.nq1:]
        u1 = u[:self.nu1]
        uj = u[self.nu1:]
                
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_u = self.B_J_R(t, q)

        P1_omega_IP1 = self.B1P1_T.T @ self.predecessor_body.B_Omega(t, q1,u1)
        P1_omega_IP1_u1 = self.B1P1_T.T @ self.predecessor_body.B_J_R(t, q1)
        pred_IB_T = self.predecessor_body.IB_T(t, q1)

        # tmp1 = cross3(self.B_kappa_R(t, q, u), B_r_P2P)
        # tmp2 = cross3(B_Omega, cross3(B_Omega, B_r_P2P))
        tmp1_u = - ax2skew(B_r_P2P) @ self.B_kappa_R_u(t, q, u)
        tmp2_u = - (ax2skew(cross3(B_Omega, B_r_P2P)) + ax2skew(B_Omega) @ ax2skew(B_r_P2P)) @ B_Omega_u
    
        kappa_P_u = self.IB_T(t, q) @ (tmp1_u + tmp2_u)

        kappa_P_u[:, : self.nu1] += self.predecessor_body.kappa_P_u(t, q1, u1, B_r_SP=self.B1_r_S1P1)

        # tmp3 = self.joint.P1_kappa_P1P2(t, qj, uj) \
        #             + cross3(self.A_B1P1.T @ self.predecessor_body.B_kappa_R(t, q1, u1), self.joint.P1_r_P1P2(t, qj))\
        #             + 2 * cross3(P1_omega_IP1, self.joint.P1_r_dot_P1P2(t, qj, uj))\
        #             + cross3(P1_omega_IP1, cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, qj)))

        tmp3_u1 = - ax2skew(self.joint.P1_r_P1P2(t, qj)) @ self.B1P1_T.T @ self.predecessor_body.B_kappa_R_u(t, q1, u1)\
                    - 2 *ax2skew(self.joint.P1_r_dot_P1P2(t, qj, uj)) @ P1_omega_IP1_u1\
                    - (ax2skew(cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, qj))) + ax2skew(P1_omega_IP1) @ ax2skew(self.joint.P1_r_P1P2(t, qj))) @ P1_omega_IP1_u1
        tmp3_uj = self.joint.P1_kappa_P1P2_u(t, qj, uj) \
                    + 2 * ax2skew(P1_omega_IP1) @ self.joint.P1_J_P1P2(t, qj)\

        kappa_P_u[:, : self.nu1] += pred_IB_T @ self.B1P1_T @ tmp3_u1
        kappa_P_u[:, self.nu1 :] += pred_IB_T @ self.B1P1_T @ tmp3_uj
        return kappa_P_u

    

