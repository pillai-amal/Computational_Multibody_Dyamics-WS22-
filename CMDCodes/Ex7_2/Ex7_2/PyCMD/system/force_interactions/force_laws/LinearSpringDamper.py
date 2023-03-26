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

    ######
    # (c)
    ######

