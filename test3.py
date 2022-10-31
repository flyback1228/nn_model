import casadi as ca
import numpy as np

a = ca.DM([[1,3],[3,4]])
b = ca.DM([[1,2],[3,4]])

print(a.elements().__hash__==b.elements().__hash__)
hash1 = a.__hash__
hash1.__eq__

c= a.elements()

print(id(a.elements()))





#b = ca.DM([[1,3],[3,4]])

a[0,1]=20
print(id(a.elements()))
a.elements()[0]=0.0
print(type(a.elements()))

print(hash1.__eq__(a.__hash__))

f = [1.0,2.0]
print(f.__eq__([2.0,2.0]))

b[0]=10
print(b)
print(a)
print(c)