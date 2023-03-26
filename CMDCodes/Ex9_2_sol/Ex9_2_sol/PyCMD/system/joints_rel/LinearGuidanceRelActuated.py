# -------------------------------------
# Computational multibody dynamics
#
# 30.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

import numpy as np

from PyCMD.system import LinearGuidanceRel

# ------ Remark: ------ 
# The subsequent implementation uses "inheritance", i.e., LinearGuidanceRelActuated is derived from the class LinearGuidanceRel.
# Loosely speaking, the call of the super constructor "super().__init__(q0=q0, u0=u0)" creates a self-object which is a LinearGuidanceRel. Then we just add the additional parts, i.e., W_tau and f_tau. We could implement this class also by copying the class LinearGuidanceRel and then add the additional parts.
# --------------------- 
class LinearGuidanceRelActuated(LinearGuidanceRel):
    def __init__(self, tau=None, q0=None, u0=None):
        """actuated linear guidance"""
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