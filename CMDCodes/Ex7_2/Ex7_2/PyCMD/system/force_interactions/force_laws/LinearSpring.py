# -------------------------------------
# Computational multibody dynamics
#
# 19.12.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

class LinearSpring:
    def __init__(self, k, g0=None):
        self.k = k      # stiffness
        self.g0 = g0    # undeformed length

    def E_pot(self, t, g):
        return 0.5 * self.k * (g - self.g0) ** 2

    def lambda_pot(self, t, g):
        return - self.k * (g - self.g0)

    def lambda_pot_g(self, t, g):
        return - self.k
