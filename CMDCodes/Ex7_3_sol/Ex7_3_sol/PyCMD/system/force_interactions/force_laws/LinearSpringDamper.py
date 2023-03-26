# -------------------------------------
# Computational multibody dynamics
#
# 19.12.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

class LinearSpringDamper:

    def __init__(self, k, c, g0=0):
        self.k = k      # spring stiffness
        self.g0 = g0    # undeformed lenght
        self.c = c      # damping ratio

    def E_pot(self, t, g):
        return 0.5 * self.k * (g - self.g0) ** 2

    def lambda_pot(self, t, g):
        return - self.k * (g - self.g0)

    def lambda_pot_g(self, t, g):
        return - self.k

    def lambda_npot(self, t, g, g_dot):
        return - self.c * g_dot

    def lambda_npot_g(self, t, g, g_dot):
        return 0

    def lambda_npot_g_dot(self, t, g, g_dot):
        return - self.c

