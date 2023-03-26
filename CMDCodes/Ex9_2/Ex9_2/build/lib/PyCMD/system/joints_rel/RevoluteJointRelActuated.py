# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

from PyCMD.system import RevoluteJointRel

class RevoluteJointRelActuated(RevoluteJointRel):
    def __init__(self, tau=None, q0=None, u0=None):
        """actuated revolute joint with rotation about e_z^P1"""
        super().__init__(q0=q0, u0=u0)
        self.ntau = 1
        
        if tau is None:
            self.tau = lambda t: np.array([0])
        else:
            self.tau = lambda t: np.array([tau(t)])

    def W_tau(self, t, q):
        return np.ones((self.nu, self.ntau))

    def f_tau(self, t, q):
        return self.tau(t)