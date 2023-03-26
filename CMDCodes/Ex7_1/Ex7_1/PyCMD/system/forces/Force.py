# -------------------------------------
# Computational multibody dynamics
#
# 19.12.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

from numpy import einsum, zeros

class Force:
    r"""Force described w.r.t. inertial frame."""

    def __init__(self,
                 force,             # force vector w.r.t. inertial frame (function of time or constant)
                 body,              # body on which force acts
                 B_r_SP=zeros(3)    # position vector from c.o.m. S of body to point of attack P of force
                 ):
        if not callable(force):
            self.force = lambda t: force
        else:
            self.force = force
        self.body = body
        self.B_r_SP = B_r_SP

    def assembler_callback(self):
        self.qDOF = self.body.qDOF
        self.uDOF = self.body.uDOF

    def E_pot(self, t, q): # potential V of the force
        ######
        # (c)
        ######
        return 

    def f_pot(self, t, q):
        ######
        # (c)
        ######
        return 

    def f_pot_q(self, t, q):
        f_q = einsum("i,ijk->jk", self.force(t), self.body.J_P_q(t, q))
        return f_q
