# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

from numpy import einsum, zeros
class Force:
    r"""Force implementation."""

    def __init__(self, force, body, B_r_SP=zeros(3)):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.body = body
        self.B_r_SP = B_r_SP

    def assembler_callback(self):
        self.qDOF = self.body.qDOF
        self.uDOF = self.body.uDOF

    def E_pot(self, t, q):
        return -(self.force(t) @ self.body.r_OP(t, q, self.B_r_SP))

    def f_pot(self, t, q):
        return self.force(t) @ self.body.J_P(t, q, self.B_r_SP)

    def f_pot_q(self, t, q):
        f_q = einsum("i,ijk->jk", self.force(t), self.body.J_P_q(t, q))
        return f_q
