# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# system class

import numpy as np

from PyCMD.system.frame.Frame import Frame

class System(object):
    """System implementation which assembles all global objects.

    """

    def __init__(self, t0=0):
        self.t0 = t0

        self.building_blocks = []

        self.contributions = []
        self.contributions.extend(["M", "Mu_q"])
        self.contributions.extend(["f_gyr", "f_gyr_q", "f_gyr_u"])
        self.contributions.extend(["f_pot", "f_pot_q"])
        self.contributions.extend(["f_npot", "f_npot_q", "f_npot_u"])
        self.contributions.extend(["q_dot", "B"])
        self.contributions.extend(["assembler_callback", "step_callback"])


        # creates empty lists for each contribution (self.M_contr = [], self.f_gyr_contr = [], ...)
        for c in self.contributions:
            setattr(self, f"_{self.__class__.__name__}__{c}_contr", [])

        self.origin = Frame()
        self.add(self.origin)

    def add(self, blocks):
        if isinstance(blocks, list) or isinstance(blocks, tuple):
            for block in blocks:
                if not block in self.building_blocks:
                    self.building_blocks.append(block)
                else:
                    raise ValueError(f"building block {str(block)} already added")
        else:
            if not blocks in self.building_blocks:
                self.building_blocks.append(blocks)
            else:
                raise ValueError(f"building block {str(blocks)} already added")


    def assemble(self):
        self.nq = 0
        self.nu = 0
        q0 = []
        u0 = []

        for block in self.building_blocks:
            block.t0 = self.t0
            for c in self.contributions:
                # if contribution is implemented as class function append building block to global contribution list
                # - c in contr.__class__.__dict__: has global class attribute c, i.e., block.c exists/makes sense
                # - callable(getattr(contr, c, None)): c is callable
                if hasattr(block, c) and callable(getattr(block, c)):
                    getattr(self, f"_{self.__class__.__name__}__{c}_contr").append(block)

            # if buliding block has position degrees of freedom address position coordinates
            if hasattr(block, "nq"):
                block.qDOF = np.arange(0, block.nq) + self.nq
                self.nq += block.nq
                q0.extend(block.q0.tolist())

            # if buliding has velocity degrees of freedom address velocity coordinates
            if hasattr(block, "nu"):
                block.uDOF = np.arange(0, block.nu) + self.nu
                self.nu += block.nu
                u0.extend(block.u0.tolist())

        self.q0 = np.array(q0)
        self.u0 = np.array(u0)

        # call assembler callback: call methods that require first an assembly of the system
        for block in self.__assembler_callback_contr:
            block.assembler_callback()
        

    #####################
    # kinematic equations
    #####################

    def q_dot(self, t, q, u):
        q_dot = np.zeros(self.nq)
        for block in self.__q_dot_contr:
            q_dot[block.qDOF] = block.q_dot(t, q[block.qDOF], u[block.uDOF])
        return q_dot

    def B(self, t, q):
        B = np.zeros((self.nq, self.nu))
        for block in self.__B_contr:
            B[block.qDOF[:, None], block.uDOF] = block.B(t, q[block.qDOF])
        return B

    #####################
    # equations of motion
    #####################
    def M(self, t, q):
        M = np.zeros((self.nu, self.nu))
        for block in self.__M_contr:
            M[block.uDOF[:, None], block.uDOF] += block.M(t, q[block.qDOF])
        return M

    def Mu_q(self, t, q, u):
        Mu_q = np.zeros((self.nu, self.nq))
        for block in self.__Mu_q_contr:
            Mu_q[block.uDOF[:, None],
                 block.qDOF] += block.Mu_q(t, q[block.qDOF], u[block.uDOF])
        return Mu_q

    def f_gyr(self, t, q, u):
        f = np.zeros(self.nu)
        for block in self.__f_gyr_contr:
            f[block.uDOF] += block.f_gyr(t, q[block.qDOF], u[block.uDOF])
        return f

    def f_gyr_q(self, t, q, u):
        f_q = np.zeros((self.nu, self.nq))
        for block in self.__f_gyr_q_contr:
            f_q[block.uDOF[:, None],
              block.qDOF] += block.f_gyr_q(t, q[block.qDOF], u[block.uDOF])
        return f_q

    def f_gyr_u(self, t, q, u):
        f_u = np.zeros((self.nu, self.nu))
        for block in self.__f_gyr_u_contr:
            f_u[block.uDOF[:, None],
              block.uDOF] += block.f_gyr_u(t, q[block.qDOF], u[block.uDOF])
        return f_u

    def E_pot(self, t, q):
        E_pot = 0
        for block in self.__f_pot_contr:
            E_pot += block.E_pot(t, q[block.qDOF])
        return E_pot

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for block in self.__f_pot_contr:
            f[block.qDOF] += block.f_pot(t, q[block.qDOF])
        return f

    def f_pot_q(self, t, q):
        f_q = np.zeros((self.nu, self.nq))
        for block in self.__f_pot_q_contr:
            f_q[block.uDOF[:, None],
                block.qDOF] += block.f_pot_q(t, q[block.qDOF])
        return f_q

    def f_npot(self, t, q, u):
        f = np.zeros(self.nu)
        for block in self.__f_npot_contr:
            f[block.uDOF] += block.f_npot(t, q[block.qDOF], u[block.uDOF])
        return f

    def f_npot_q(self, t, q, u):
        f_q = np.zeros((self.nu, self.nq))
        for block in self.__f_npot_q_contr:
            f_q[block.uDOF[:, None],
                block.qDOF] += block.f_npot_q(t, q[block.qDOF], u[block.uDOF])
        return f_q

    def f_npot_u(self, t, q, u):
        f_u = np.zeros((self.nu, self.nu))
        for block in self.__f_npot_u_contr:
            f_u[block.uDOF[:, None],
                block.uDOF]+= block.f_npot_u(t, q[block.qDOF], u[block.uDOF])
        return f_u

    def h(self, t, q, u):
        return self.f_pot(t, q) + self.f_npot(t, q, u) - self.f_gyr(t, q, u)

    def h_q(self, t, q, u):
        return (
            self.f_pot_q(t, q)
            + self.f_npot_q(t, q, u)
            - self.f_gyr_q(t, q, u)
        )

    def h_u(self, t, q, u):
        return self.f_npot_u(t, q, u) - self.f_gyr_u(t, q, u)

