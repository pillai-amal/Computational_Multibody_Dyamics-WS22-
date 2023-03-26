# -------------------------------------
# Computational multibody dynamics
# Exercise 3.3 - Solution
#
# 31.10.22 - Dr.-Ing. G. Capobianco
# -------------------------------------

# -------------------------------------
# c)
print('---\n c)')

n = 100
f = 0
for i in range(1, n + 1):
    f = f + i

print(f'f(100) = {f}')

## Remarks: 
# - 'range' uses half open intervals, i.e., range(1, 101) returns the list [1, ..., 100]. 
# - 'f = f + i' can compactly be written as 'f += i'

# -------------------------------------
# d)
print('---\n d)')

# define the function f
def f(n):
    ff = 0
    for i in range(1, n + 1):
        ff += i
    return ff

# print f(75)
print(f'f(75) = {f(75)}')

# -------------------------------------
# e)
print('---\n e)')

# import NumPy and name it 'np'
import numpy as np

# define the function f using normal syntax
def f1(n):
    return np.sum(np.arange(1, n + 1))

# define the function f using lambda function syntax (shorter, good for simple functions)
f2 = lambda n: np.sum(np.arange(1, n + 1))

# print f(75)
print(f'f(75) = {f1(75)}')
print(f'f(75) = {f2(75)}')

# -------------------------------------
# f)

# form the module 'matplotlib' import the submodule 'pyplot' and name it 'plt'
from matplotlib import pyplot as plt

# in order to plot 'f' we need a function which returns a list f_list(n) = [f(1), ..., f(n)]
def f_list(n):
    ff = [] # empty list
    for i in range(1, n+1):
        ff.append(f2(i)) # appends a new entry 'f(i)' to the list
    return ff

n = 20
n_list = np.arange(1, n + 1)

# create the plot
plt.plot(n_list, f_list(n))
# name axes
plt.xlabel('n')
plt.ylabel('f')
# show plot
plt.show()

# -------------------------------------
# g)
print('---\n g)')

# we can use 'np' since it has already been imported.

A = np.array([[1, 2, 3],
              [0, 1, 2],
              [0, 0, 3]])

# create a 3x3-matix filled with ones
B = np.ones((3, 3))
B[1] = 2 # set values of second row to 2
B[2, :] = 3 # set values of third row to 3, this is equivalent to 'B[2s] = 3'

# create vector b using arange
b = np.arange(1, 4)

# computations with the above matrices
x = A @ b
print(f'Ab = {x}')

x = b @ A
print(f'b_transp A = {x}')

x = A @ B
print(f'AB = {x}')

x = A.T @ B @ b
print(f'A_transp Bb = {x}')

x = np.linalg.solve(A, b)
print(f'A_inv b = {x}')