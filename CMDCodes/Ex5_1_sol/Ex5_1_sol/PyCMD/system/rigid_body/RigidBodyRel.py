# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

from PyCMD.math import cross3, ax2skew, ax2skew_a

class RigidBodyRel:
    def __init__(self, 
                m, 
                B_theta_S, 
                joint, 
                predecessor_body, 
                B1_r_S1P1=np.zeros(3),
                A_B1P1=np.eye(3),
                B2_r_S2P2=np.zeros(3),
                A_B2P2=np.eye(3)
                ):
        self.m = m
        self.B_theta_S = B_theta_S

        self.predecessor_body = predecessor_body

        self.joint = joint

        self.B1_r_S1P1 = B1_r_S1P1
        self.A_B1P1 = A_B1P1
        self.B2_r_S2P2 = B2_r_S2P2
        self.A_P2B2 = A_B2P2.T

        self.is_assembled = False

    def assembler_callback(self):
        if not self.joint.is_assembled:
            raise RuntimeError("Joint is not assembled; maybe not added to the model.")

        if not self.predecessor_body.is_assembled:
            raise RuntimeError(
                "Predecessor body is not assembled; maybe not added to the model."
            )

        self.qDOF = np.concatenate([self.predecessor_body.qDOF, self.joint.qDOF])
        self.q0 = np.concatenate([self.predecessor_body.q0, self.joint.q0])
        self.__nq = len(self.qDOF)
        self.nq1 = len(self.predecessor_body.qDOF)

        self.uDOF = np.concatenate([self.predecessor_body.uDOF, self.joint.uDOF])
        self.u0 = np.concatenate([self.predecessor_body.u0, self.joint.u0])
        self.__nu = len(self.uDOF)
        self.nu1 = len(self.predecessor_body.uDOF)

        self.is_assembled = True

    #####################
    # equations of motion
    #####################

    def M(self, t, q):
        J_S = self.J_P(t, q)
        J_R = self.B_J_R(t, q)
        return self.m * J_S.T @ J_S + J_R.T @ self.B_theta_S @ J_R
        
    def Mu_q(self, t, q, u):
        J_S = self.J_P(t, q)
        J_R = self.B_J_R(t, q)
        J_S_q = self.J_P_q(t, q)
        J_R_q = self.B_J_R_q(t, q)

        Mu_q = (
            np.einsum("ijl,ik,k->jl", J_S_q, J_S, self.m * u)
            + np.einsum("ij,ikl,k->jl", J_S, J_S_q, self.m * u)
            + np.einsum("ijl,ik,k->jl", J_R_q, self.B_theta_S @ J_R, u)
            + np.einsum("ij,jkl,k->il", J_R.T @ self.B_theta_S, J_R_q, u)
        )

        return Mu_q

    def f_gyr(self, t, q, u):
        Omega = self.B_Omega(t, q, u)
        return self.m * self.J_P(t, q).T @ self.kappa_P(t, q, u) \
                + self.B_J_R(t, q).T @ (self.B_theta_S @ self.B_kappa_R(t, q, u)
                                        + cross3(Omega, self.B_theta_S @ Omega))

    def f_gyr_q(self, t, q, u):
        Omega = self.B_Omega(t, q, u)
        Omega_q = self.B_Omega_q(t, q, u)
        tmp1 = self.B_theta_S @ self.B_kappa_R(t, q, u)
        tmp1_q = self.B_theta_S @ self.B_kappa_R_q(t, q, u)
        tmp2 = cross3(Omega, self.B_theta_S @ Omega)
        tmp2_q = (ax2skew(Omega) @ self.B_theta_S - ax2skew(self.B_theta_S @ Omega)) @ Omega_q

        f_gyr_q = (
            np.einsum("jik,j->ik", self.J_P_q(t, q), self.m * self.kappa_P(t, q, u))
            + self.m * self.J_P(t, q).T @ self.kappa_P_q(t, q, u)
            + np.einsum("jik,j->ik", self.J_P_q(t, q), tmp1 + tmp2)
            + self.B_J_R(t, q).T @ (tmp1_q + tmp2_q)
        )

        return f_gyr_q

    def f_gyr_u(self, t, q, u):
        Omega = self.B_Omega(t, q, u)
        Omega_u = self.B_J_R(t, q)
        tmp1_u = self.B_theta_S @ self.B_kappa_R_u(t, q, u)
        tmp2_u = (ax2skew(Omega) @ self.B_theta_S - ax2skew(self.B_theta_S @ Omega)) @ Omega_u

        f_gyr_u = self.m * self.J_P(t, q).T @ self.kappa_P_u(t, q, u) + self.B_J_R(
            t, q
        ).T @ (tmp1_u + tmp2_u)
        
        return f_gyr_u

    
    #########################################
    # rigid body kinematics
    #########################################

    def A_IB(self, t, q):
        return self.predecessor_body.A_IB(t, q[:self.nq1])\
                @ self.A_B1P1\
                    @ self.joint.A_P1P2(t, q[self.nq1:])\
                        @ self.A_P2B2

    def A_IB_q(self, t, q):
        A_IB_q = np.zeros((3, 3, self.__nq))
        A_IB_q[:, :, : self.nq1] = np.einsum(
            "ijk,jl->ilk", self.predecessor_body.A_IB_q(t, q[:self.nq1]),
                                    self.A_B1P1\
                                        @ self.joint.A_P1P2(t, q[self.nq1:])\
                                            @ self.A_P2B2
        )
        A_IB_q[:, :, self.nq1 :] = np.einsum(
            "ij,jkl,km->iml", self.predecessor_body.A_IB(t, q[:self.nq1])\
                                    @ self.A_B1P1,
                                        self.joint.A_P1P2_q(t, q[self.nq1:]),
                                            self.A_P2B2
        )

        return A_IB_q

    def r_OP(self, t, q, B_r_SP=np.zeros(3)):
        return (
            self.predecessor_body.r_OP(t, q[:self.nq1], B_r_SP=self.B1_r_S1P1)
            + self.predecessor_body.A_IB(t, q[:self.nq1]) @ self.A_B1P1 @ self.joint.P1_r_P1P2(t, q[self.nq1:])
            + self.A_IB(t, q) @ (B_r_SP - self.B2_r_S2P2)
        )

    def v_P(self, t, q, u, B_r_SP=np.zeros(3)):
        v_P2 = self.predecessor_body.v_P(t, q[:self.nq1], u[:self.nu1], B_r_SP=self.B1_r_S1P1) \
                + self.predecessor_body.A_IB(t, q[:self.nq1]) @ self.A_B1P1 @ \
                    (self.joint.P1_v_P1P2(t, q[self.nq1:], u[self.nu1:])
                + cross3(self.A_B1P1.T @ self.predecessor_body.B_Omega(t, q[:self.nq1], u[:self.nu1]), self.joint.P1_r_P1P2(t, q[self.nq1:])))
        v_P2P = self.A_IB(t, q) @ cross3(self.B_Omega(t, q, u), B_r_SP - self.B2_r_S2P2)
        return v_P2 + v_P2P

    def J_P(self, t, q, B_r_SP=np.zeros(3)):
           
        pred_A_IB = self.predecessor_body.A_IB(t, q[:self.nq1])
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        
        J_P = - self.A_IB(t, q) @ ax2skew(B_r_P2P) @ self.B_J_R(t, q)
        J_P[:, :self.nu1] += self.predecessor_body.J_P(t, q[:self.nq1], B_r_SP=self.B1_r_S1P1) \
            - pred_A_IB @ self.A_B1P1 @ ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:])) @ self.A_B1P1.T @ self.predecessor_body.B_J_R(t, q[:self.nq1])
        J_P[:, self.nu1:] += pred_A_IB @ self.A_B1P1 @ self.joint.P1_J_P1P2(t, q[self.nq1:])
        return J_P

    def J_P_q(self, t, q, B_r_SP=np.zeros(3)):
        
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        pred_A_IB = self.predecessor_body.A_IB(t, q[:self.nq1])
        pred_A_IB_q = self.predecessor_body.A_IB_q(t, q[:self.nq1])
        pred_B_J_R = self.predecessor_body.B_J_R(t, q[:self.nq1])
        
        J_P_q = np.einsum(
            "ij,jkl->ikl",
            - self.A_IB(t, q) @ ax2skew(B_r_P2P), self.B_J_R_q(t, q)
        )
        J_P_q -= np.einsum(
            "ijk,jl->ilk",
            self.A_IB_q(t, q), ax2skew(B_r_P2P) @ self.B_J_R(t, q)
        )
        J_P_q[:,
              : self.nu1,
              : self.nq1] += self.predecessor_body.J_P_q(t,
                                                         q[:self.nq1],
                                                         B_r_SP=self.B1_r_S1P1)
        J_P_q[:,
              : self.nu1,
              : self.nq1] -= np.einsum(
                "ij,jlk->ilk",
              pred_A_IB @ self.A_B1P1 @ ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:])) @ self.A_B1P1.T , self.predecessor_body.B_J_R_q(t, q[:self.nq1]))

        J_P_q[:,
              : self.nu1,
              : self.nq1] -= np.einsum(
                "ijk,jl->ilk",
              pred_A_IB_q, self.A_B1P1 @ ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:])) @ self.A_B1P1.T @ pred_B_J_R)

        J_P_q[:,
              : self.nu1,
               self.nq1 :] -= np.einsum(
                "ijk,jl->ilk",
              pred_A_IB @ self.A_B1P1 @ ax2skew_a() @ self.joint.P1_r_P1P2_q(t, q[self.nq1:]), self.A_B1P1.T @ pred_B_J_R)

        J_P_q[:,
              self.nu1 :,
              : self.nq1] += np.einsum(
            "ijk,jl->ilk", pred_A_IB_q,
            self.A_B1P1 @ self.joint.P1_J_P1P2(t, q[self.nq1:])
        )
        J_P_q[:,
              self.nu1 :,
              self.nq1 :] += np.einsum(
            "ij,jlk->ilk",
            pred_A_IB @ self.A_B1P1,
            self.joint.P1_J_P1P2_q(t, q[self.nq1:])
        )

        return J_P_q

    def kappa_P(self, t, q, u, B_r_SP=np.zeros(3)):
        # return self.a_P(t, q, u, np.zeros(self.__nu), K_r_SP=K_r_SP)

        B_r_P2P = B_r_SP - self.B2_r_S2P2

        P1_omega_IP1 = self.A_B1P1.T @ self.predecessor_body.B_Omega(t, q[:self.nq1],u[:self.nu1])

        kappa_P = self.predecessor_body.kappa_P(t, q[:self.nq1], u[:self.nu1], B_r_SP=self.B1_r_S1P1) \
            + self.predecessor_body.A_IB(t, q[:self.nq1]) @ self.A_B1P1\
                    @ (self.joint.P1_kappa_P1P2(t, q[self.nq1:], u[self.nu1:]) \
                        + cross3(self.A_B1P1.T @ self.predecessor_body.B_kappa_R(t, q[:self.nq1], u[:self.nu1]), self.joint.P1_r_P1P2(t, q[self.nq1:]))\
                        + cross3(2 * P1_omega_IP1, self.joint.P1_v_P1P2(t, q[self.nq1:], u[self.nq1:]))\
                    + cross3(P1_omega_IP1, cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, q[self.nq1:]))))

        kappa_P += self.A_IB(t, q) @ (
            cross3(self.B_kappa_R(t, q, u), B_r_P2P)
            + cross3(self.B_Omega(t, q, u), cross3(self.B_Omega(t, q, u), B_r_P2P))
        )

        return kappa_P


    def kappa_P_q(self, t, q, u, B_r_SP=np.zeros(3)):
        
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_q = self.B_Omega_q(t, q, u)
        P1_omega_IP1 = self.A_B1P1.T @ self.predecessor_body.B_Omega(t, q[:self.nq1],u[:self.nu1])
        P1_omega_IP1_q1 = self.A_B1P1.T @ self.predecessor_body.B_Omega_q(t, q[:self.nq1], u[:self.nu1])
        pred_kappa_P_R = self.predecessor_body.B_kappa_R(t, q[:self.nq1], u[:self.nu1])
        pred_A_IB = self.predecessor_body.A_IB(t, q[:self.nq1])
   

        tmp1 = cross3(self.B_kappa_R(t, q, u), B_r_P2P)
        tmp1_q = -ax2skew(B_r_P2P) @ self.B_kappa_R_q(t, q, u)
        tmp2 = cross3(B_Omega, cross3(B_Omega, B_r_P2P))
        tmp2_q = (
            -(ax2skew(cross3(B_Omega, B_r_P2P)) + ax2skew(B_Omega) @ ax2skew(B_r_P2P))
            @ B_Omega_q)

        kappa_P_q = np.einsum("ijk,j->ik", self.A_IB_q(t, q), tmp1 + tmp2)
        + self.A_IB(t, q) @ (tmp1_q + tmp2_q)
        

        kappa_P_q[:, : self.nq1] += self.predecessor_body.kappa_P_q(t, q[:self.nq1], u[:self.nu1])

        tmp3 = self.joint.P1_kappa_P1P2(t, q[self.nq1:], u[self.nu1:]) \
                    + cross3(self.A_B1P1.T @ pred_kappa_P_R, self.joint.P1_r_P1P2(t, q[self.nq1:]))\
                    + 2 * cross3(P1_omega_IP1, self.joint.P1_v_P1P2(t, q[self.nq1:], u[self.nq1:]))\
                    + cross3(P1_omega_IP1, cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, q[self.nq1:])))

        tmp3_q1 = - ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:])) @ self.A_B1P1.T @ self.predecessor_body.B_kappa_R_q(t, q[:self.nq1], u[:self.nu1])\
                  - 2 * ax2skew(self.joint.P1_v_P1P2(t, q[self.nq1:], u[self.nq1:])) @ P1_omega_IP1_q1\
                  - (ax2skew(cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, q[self.nq1:]))) + ax2skew(P1_omega_IP1) @ ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:]))) @ P1_omega_IP1_q1
                  
        tmp3_qj = self.joint.P1_kappa_P1P2_q(t, q[self.nq1:], u[self.nu1:]) \
                    + ax2skew(self.A_B1P1.T @ pred_kappa_P_R) @ self.joint.P1_r_P1P2_q(t, q[self.nq1:])\
                    + 2 * ax2skew(P1_omega_IP1) @ self.joint.P1_v_P1P2_q(t, q[self.nq1:], u[self.nq1:])\
                    + ax2skew(P1_omega_IP1) @ ax2skew(P1_omega_IP1) @ self.joint.P1_r_P1P2_q(t, q[self.nq1:])

        kappa_P_q[:, : self.nq1] +=  np.einsum("ijk,j->ik", self.predecessor_body.A_IB_q(t, q[:self.nq1]),
                        self.A_B1P1 @ tmp3) + pred_A_IB@ self.A_B1P1 @ tmp3_q1
        
        kappa_P_q[:, self.nq1 :] += pred_A_IB @ self.A_B1P1 @ tmp3_qj
        return kappa_P_q

    def kappa_P_u(self, t, q, u, B_r_SP=np.zeros(3)):
                     
        B_r_P2P = B_r_SP - self.B2_r_S2P2
        B_Omega = self.B_Omega(t, q, u)
        B_Omega_u = self.B_J_R(t, q)

        P1_omega_IP1 = self.A_B1P1.T @ self.predecessor_body.B_Omega(t, q[:self.nq1],u[:self.nu1])
        P1_omega_IP1_u1 = self.A_B1P1.T @ self.predecessor_body.B_J_R(t, q[:self.nq1])
        pred_A_IB = self.predecessor_body.A_IB(t, q[:self.nq1])

        # tmp1 = cross3(self.B_kappa_R(t, q, u), B_r_P2P)
        # tmp2 = cross3(B_Omega, cross3(B_Omega, B_r_P2P))
        tmp1_u = - ax2skew(B_r_P2P) @ self.B_kappa_R_u(t, q, u)
        tmp2_u = - (ax2skew(cross3(B_Omega, B_r_P2P)) + ax2skew(B_Omega) @ ax2skew(B_r_P2P)) @ B_Omega_u
    
        kappa_P_u = self.A_IB(t, q) @ (tmp1_u + tmp2_u)

        kappa_P_u[:, : self.nu1] += self.predecessor_body.kappa_P_u(t, q[:self.nq1], u[:self.nu1], B_r_SP=self.B1_r_S1P1)

        # tmp3 = self.joint.P1_kappa_P1P2(t, q[self.nq1:], u[self.nu1:]) \
        #             + cross3(self.A_B1P1.T @ self.predecessor_body.B_kappa_R(t, q[:self.nq1], u[:self.nu1]), self.joint.P1_r_P1P2(t, q[self.nq1:]))\
        #             + 2 * cross3(P1_omega_IP1, self.joint.P1_v_P1P2(t, q[self.nq1:], u[self.nq1:]))\
        #             + cross3(P1_omega_IP1, cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, q[self.nq1:])))

        tmp3_u1 = - ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:])) @ self.A_B1P1.T @ self.predecessor_body.B_kappa_R_u(t, q[:self.nq1], u[:self.nu1])\
                    - 2 *ax2skew(self.joint.P1_v_P1P2(t, q[self.nq1:], u[self.nq1:])) @ P1_omega_IP1_u1\
                    - (ax2skew(cross3(P1_omega_IP1, self.joint.P1_r_P1P2(t, q[self.nq1:]))) + ax2skew(P1_omega_IP1) @ ax2skew(self.joint.P1_r_P1P2(t, q[self.nq1:]))) @ P1_omega_IP1_u1
        tmp3_uj = self.joint.P1_kappa_P1P2_u(t, q[self.nq1:], u[self.nu1:]) \
                    + 2 * ax2skew(P1_omega_IP1) @ self.joint.P1_J_P1P2(t, q[self.nq1:])\

        kappa_P_u[:, : self.nu1] += pred_A_IB @ self.A_B1P1 @ tmp3_u1
        kappa_P_u[:, self.nu1 :] += pred_A_IB @ self.A_B1P1 @ tmp3_uj
        return kappa_P_u

    def B_Omega(self, t, q, u):
        P1_Omega = self.A_B1P1.T\
            @ self.predecessor_body.B_Omega(t, q[:self.nq1], u[:self.nu1])\
            + self.joint.P1_omega_P1P2(t, q[self.nq1:], u[self.nu1:])
        return ( self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2 ).T @ P1_Omega

    def B_Omega_q(self, t, q, u):
        
        P1_Omega_q = np.einsum("ji,jl->il",
                               self.A_B1P1,
                               self.predecessor_body.B_Omega_q(t,
                                                               q[:self.nq1],
                                                               u[:self.nu1]))\
        + self.joint.P1_omega_P1P2_q(t, q[self.nq1:], u[self.nu1:])
        P1_Omega = self.A_B1P1.T\
            @ self.predecessor_body.B_Omega(t, q[:self.nq1], u[:self.nu1])\
            + self.joint.P1_omega_P1P2(t, q[self.nq1:], u[self.nu1:])
        A_B2P1_q = np.einsum("ijl,jk->kil", self.joint.A_P1P2_q(t, q[self.nq1:]),
                             self.A_P2B2)
        B_Omega_q = np.einsum("ijk,j->ik", A_B2P1_q, P1_Omega)
        + (self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2).T @ P1_Omega_q
 
        return B_Omega_q

    def B_J_R(self, t, q):

        J_R = np.zeros((3, self.__nu))
        A_B2P1 = ( self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2 ).T
        J_R[:, : self.nu1] = A_B2P1 @ self.A_B1P1.T\
                                    @ self.predecessor_body.B_J_R(t, q[:self.nq1])
        J_R[:, self.nu1 :] = A_B2P1 @ self.joint.P1_J_R_P1P2(t, q[self.nq1:])

        return J_R

    def B_J_R_q(self, t, q):
       
        B_J_R_q = np.zeros((3, self.__nu, self.__nq))

        A_B2P1 = ( self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2 ).T
        # transpose via einsum li instead of il.T
        A_B2P2_q = np.einsum("ijk,jl->lik",
                             self.joint.A_P1P2_q(t, q[self.nq1:]), self.A_P2B2)

        B_J_R_q[:, : self.nu1, : self.nq1] = np.einsum(
            "ij,jkl->ikl", A_B2P1 @ self.A_B1P1.T,
            self.predecessor_body.B_J_R_q(t, q[:self.nq1])
        )
        B_J_R_q[:, : self.nu1, self.nq1 :] = np.einsum(
            "ijk,jl->ilk", A_B2P2_q,
            self.A_B1P1.T @ self.predecessor_body.B_J_R(t, q[:self.nq1])
        )
        B_J_R_q[:, self.nu1 :, self.nq1 :] = np.einsum(
            "ijk,jl->ilk", A_B2P2_q, self.joint.P1_J_R_P1P2(t, q[self.nq1:])
        ) + np.einsum("ij,jkl->ikl", A_B2P1, self.joint.P1_J_R_P1P2_q(t, q[self.nq1:])
        )

        return B_J_R_q

    def B_kappa_R(self, t, q, u):
        # return self.K_Psi(t, q, u, np.zeros(self.__nu))

        A_B2P1 = ( self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2 ).T
        P1_kappa_R = self.A_B1P1.T\
            @ self.predecessor_body.B_kappa_R(t, q[:self.nq1], u[:self.nu1])\
                + self.joint.P1_kappa_R_P1P2(t, q[self.nq1:], u[self.nu1:])\
                + cross3(self.A_B1P1.T @ self.predecessor_body.B_Omega(t, q[:self.nq1], u[:self.nu1]), self.joint.P1_omega_P1P2(t, q[self.nq1:], u[self.nu1:]))
        return A_B2P1 @ P1_kappa_R

    def B_kappa_R_q(self, t, q, u):
        B_kappa_R_q = np.zeros((3, self.__nq))
        
        A_B2P1 = ( self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2 ).T
        pred_B_Omega = self.predecessor_body.B_Omega(t, q[:self.nq1], u[:self.nu1])

        P1_kappa_R = self.A_B1P1.T\
            @ self.predecessor_body.B_kappa_R(t, q[:self.nq1], u[:self.nu1])\
                + self.joint.P1_kappa_R_P1P2(t, q[self.nq1:], u[self.nu1:])\
                + cross3(self.A_B1P1.T @ pred_B_Omega , self.joint.P1_omega_P1P2(t, q[self.nq1:], u[self.nu1:]))
            
        B_kappa_R_q[:, : self.nq1] = A_B2P1\
            @ (self.A_B1P1.T @ self.predecessor_body.B_kappa_R_q(t, q[:self.nq1], u[:self.nu1]) \
                - ax2skew(self.joint.P1_omega_P1P2(t, q[self.nq1:], u[self.nu1:])) @  self.A_B1P1.T @ self.predecessor_body.B_Omega_q(t, q[:self.nq1], u[:self.nu1]))
        B_kappa_R_q[:, self.nq1 :] = A_B2P1\
            @ (self.joint.P1_kappa_R_P1P2_q(t, q[self.nq1:], u[self.nu1:]) + ax2skew(self.A_B1P1.T @ pred_B_Omega) @ self.joint.P1_omega_P1P2_q(t, q[self.nq1:], u[self.nu1:]))\
            + np.einsum("ji,kjl,k->il", self.A_P2B2, self.joint.A_P1P2_q(t, q[self.nq1:]), P1_kappa_R)
            

        return B_kappa_R_q

    def B_kappa_R_u(self, t, q, u):
        A_B2P1 = ( self.joint.A_P1P2(t, q[self.nq1:]) @ self.A_P2B2 ).T
        B_kappa_R_u = np.zeros((3, self.__nu))

        B_kappa_R_u[:, : self.nu1] = A_B2P1\
            @ (self.A_B1P1.T @ self.predecessor_body.B_kappa_R_u(t, q[:self.nq1], u[:self.nu1])\
                + ax2skew(self.joint.P1_omega_P1P2(t, q[self.nq1:], u[self.nu1:])) @ self.A_B1P1.T @ self.predecessor_body.B_J_R(t, q[:self.nq1]) )
        B_kappa_R_u[:, self.nu1 :] = A_B2P1\
            @ (self.joint.P1_kappa_R_P1P2_u(t, q[self.nq1:], u[self.nu1:]) + ax2skew(self.A_B1P1.T @ self.predecessor_body.B_Omega(t, q[:self.nq1], u[:self.nu1])) @ self.joint.P1_J_R_P1P2(t, q[self.nq1:]) )
        return B_kappa_R_u

