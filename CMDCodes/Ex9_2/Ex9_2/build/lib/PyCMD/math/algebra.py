# -------------------------------------
# Computational multibody dynamics
#
# 14.11.22 - Dr.-Ing. G. Capobianco
# -------------------------------------
# Algebraic operations

import numpy as np

e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])


def ei(i: int) -> np.ndarray:
    """Retuns the i-th Cartesian basis vector.
    With i=0: e1, i=1: e2, i=2: e3, i=3: e1, etc."""
    return np.roll(e1, i)

def norm(a: np.ndarray) -> float:
    """Euclidean norm of an array of arbitrary length."""
    return np.sqrt(a @ a)

def ax2skew(a: np.ndarray) -> np.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
    # fmt: on


def skew2ax(A: np.ndarray) -> np.ndarray:
    """Computes the axial vector from a skew symmetric 3x3 matrix."""
    assert A.shape == (3, 3)
    # fmt: off
    return 0.5 * np.array([A[2, 1] - A[1, 2], 
                           A[0, 2] - A[2, 0], 
                           A[1, 0] - A[0, 1]], dtype=A.dtype)
    # fmt: on


def ax2skew_a():
    """
    Partial derivative of the `ax2skew` function with respect to its argument.

    Note:
    -----
    This is a constant 3x3x3 ndarray."""
    A = np.zeros((3, 3, 3), dtype=float)
    A[1, 2, 0] = -1
    A[2, 1, 0] = 1
    A[0, 2, 1] = 1
    A[2, 0, 1] = -1
    A[0, 1, 2] = -1
    A[1, 0, 2] = 1
    return A

def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector product of two 3D vectors."""
    assert a.size == 3
    assert b.size == 3
    # fmt: off
    return np.array([a[1] * b[2] - a[2] * b[1], \
                     a[2] * b[0] - a[0] * b[2], \
                     a[0] * b[1] - a[1] * b[0] ])
    # fmt: on

