import casadi as ca

x=ca.MX.sym('x')
y=ca.MX.sym('y')

f = ca.Function('f',[x,y],[x+y**2])
print(f)

j1 = ca.Function('j',[x,y],[ca.jacobian(f(x,y),x)])


print(j1)