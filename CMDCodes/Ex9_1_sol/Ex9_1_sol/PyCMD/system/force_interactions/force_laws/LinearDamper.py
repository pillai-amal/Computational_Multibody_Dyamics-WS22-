# -------------------------------------
# Computational multibody dynamics
#
# 19.12.22 - Dr.-Ing. G. Capobianco
# -------------------------------------


class LinearDamper:
    def __init__(self, c):
        self.c = c

    def lambda_npot(self, t, g, g_dot):
        return - self.c * g_dot

    def lambda_npot_g(self, t, g, g_dot):
        return 0

    def lambda_npot_g_dot(self, t, g, g_dot):
        return - self.c
