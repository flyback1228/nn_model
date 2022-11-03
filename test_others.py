import numpy as np
import casadi as ca


l = [1.0,2.0,3.0,4.0]

ca_a = ca.DM([[1,2],[3,4]])

a = np.asarray(l)

a[2]=4.0

print(l)
print(a)

import array
A = array.array('i', range(10))

np_A = np.asarray(A)
np_A[4] = 555
print(np_A)
print(A)

print(type(ca_a.elements()))



print(b)