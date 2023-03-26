# -------------------------------------
# Computational multibody dynamics
#
# 19.12.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np
from numpy.linalg import norm

class LinearMotor:
    r"""Linear motor between a point P1 on body1 and P2 on body2."""

    def __init__(self,
                 body1,                 # body1 
                 body2,                 # body2 
                 F=lambda t:np.zeros(1),# motor force (F>0 pulls P1 and P2 closer together)
                 B_r_SP1=np.zeros(3),   # position vector from c.o.m. S1 of body 1 to point of attack P1 the interaction
                 B_r_SP2=np.zeros(3)    # position vector from c.o.m. S2 of body 2 to point of attack P2 the interaction
                 ):
        self.ntau = 1
        self.tau = F
        self.body1 = body1
        self.body2 = body2
        self.B_r_SP1 = B_r_SP1
        self.B_r_SP2 = B_r_SP2


    def assembler_callback(self):
        # read out connectivity of this force interaction
        self.qDOF = np.concatenate([self.body1.qDOF, self.body2.qDOF])
        self.nq1 = len(self.body1.qDOF)

        self.uDOF = np.concatenate([self.body1.uDOF, self.body2.uDOF])
        self.nu1 = len(self.body1.uDOF)
        self.__nu = len(self.uDOF)

    #####################
    # force contributions
    #####################

    def W_tau(self, t, q):
        return - self.w(t, q).reshape((self.__nu, self.ntau))

    def f_tau(self, t, q):
        return - self.tau(t)[0] * self.w(t, q)

    def f_tau_q(self, t, q):
        return - self.tau(t)[0] * self.w_q(t, q)

    ######################
    # kinematic quantities
    ######################

    def n(self, t, q):
        r_OP1 = self.body1.r_OP(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        r_OP2 = self.body2.r_OP(t, q[self.nq1:], B_r_SP=self.B_r_SP2)
        return (r_OP2 - r_OP1) / norm(r_OP2 - r_OP1)

    def n_q(self, t, q):
        r_OP1 = self.body1.r_OP(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        r_OP2 = self.body2.r_OP(t, q[self.nq1:], B_r_SP=self.B_r_SP2)

        r_OP1_q = self.body1.r_OP_q(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        r_OP2_q = self.body2.r_OP_q(t, q[self.nq1:], B_r_SP=self.B_r_SP2)

        r_P1P2 = r_OP2 - r_OP1

        g = norm(r_P1P2)

        tmp = np.outer(r_P1P2, r_P1P2) / (g**3)
        n_q1 = -r_OP1_q / g + tmp @ r_OP1_q
        n_q2 = r_OP2_q / g - tmp @ r_OP2_q

        return n_q1, n_q2

    def g(self, t, q):
        r_OP1 = self.body1.r_OP(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        r_OP2 = self.body2.r_OP(t, q[self.nq1:], B_r_SP=self.B_r_SP2)
        return norm(r_OP2 - r_OP1)

    def g_q(self, t, q):
        n = self.n(t, q)

        r_OP1_q1 = self.body1.r_OP_q(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        r_OP2_q2 = self.body2.r_OP_q(t, q[self.nq1:], B_r_SP=self.B_r_SP2)

        return np.hstack((-n @ r_OP1_q1, n @ r_OP2_q2))

    def g_dot(self, t, q, u):
        v_P1 = self.body1.v_P(t, q[:self.nq1], u[:self.nu1], B_r_SP=self.B_r_SP1)
        v_P2 = self.body2.v_P(t, q[self.nq1:], u[self.nu1:], B_r_SP=self.B_r_SP2)
        return self.n(t, q) @ (v_P2 - v_P1)

    def g_dot_q(self, t, q, u):
        n = self.n(t, q)
        n_q1, n_q2 = self.n_q(t, q)

        v_P1 = self.body1.v_P(t, q[:self.nq1], u[:self.nu1], B_r_SP=self.B_r_SP1)
        v_P2 = self.body2.v_P(t, q[self.nq1:], u[self.nu1:], B_r_SP=self.B_r_SP2)

        v_P1_q1 = self.body1.v_P_q(t, q[:self.nq1], u[:self.nu1], B_r_SP=self.B_r_SP1)
        v_P2_q2 = self.body2.v_P_q(t, q[self.nq1:], u[self.nu1:], B_r_SP=self.B_r_SP2)
        
        return np.hstack((-n @ v_P1_q1 - v_P1 @ n_q1, n @ v_P2_q2 + v_P2 @ n_q2))

    def g_dot_u(self, t, q, u):
        n = self.n(t, q)

        J_P1 = self.body1.J_P(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        J_P2 = self.body2.J_P(t, q[self.nq1:], B_r_SP=self.B_r_SP2)
        
        return np.hstack((-n @ J_P1, n @ J_P2))

    def w(self, t, q):
        n = self.n(t, q)

        J_P1 = self.body1.J_P(t, q[:self.nq1], B_r_SP=self.B_r_SP1)
        J_P2 = self.body2.J_P(t, q[self.nq1:], B_r_SP=self.B_r_SP2)

        return np.hstack((-n @ J_P1, n @ J_P2))
    
    def w_q(self, t, q):
        w_q = np.zeros((len(self.uDOF), len(self.qDOF)))
        n_q1, n_q2 = self.n_q(t, q)
        w_q[:self.nu1, :self.nq1] = np.einsum("ij,il->lj", n_q1, self.body1.J_P(t, q[:self.nq1], B_r_SP=self.B_r_SP1)) + np.einsum("i,ikj->kj", self.n(t, q), self.body1.J_P_q(t, q[:self.nq1], B_r_SP=self.B_r_SP1))
        w_q[:self.nu1, self.nq1:] = self.body1.J_P(t, q[:self.nq1], B_r_SP=self.B_r_SP1).T @ n_q2
        w_q[self.nu1:, :self.nq1] = self.body2.J_P(t, q[self.nq1:], B_r_SP=self.B_r_SP2).T @ n_q1
        w_q[self.nu1:, self.nq1:] = self.body2.J_P(t, q[self.nq1:], B_r_SP=self.B_r_SP2).T @ n_q2 + np.einsum("i,ikj->kj", self.n(t, q), self.body2.J_P_q(t, q[self.nq1:], B_r_SP=self.B_r_SP2))
        return w_q

    