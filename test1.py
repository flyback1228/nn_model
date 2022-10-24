import casadi as ca
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
from jax_casadi_callback import JaxCasadiCallbackSISO
import tensorflow as tf

horizon = 50
nx = 8
nu = 2
dt = 0.05
x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)

option = {}
option['max_iter']=10000


# Declare model variables
x1 = ca.MX.sym('x1',10,1)
tf_model= tf.keras.models.load_model('output/disk.h5')


a = np.array([[1, 1, -1.5, 1, -1, 1, 0, 0,0,0]],dtype=np.float32)
b = tf_model(a)
print(b)

f = jax2tf.call_tf(lambda x: np.transpose(tf_model.predict(tf.transpose(x))))
print(f(np.transpose(a)))

jax_model = JaxCasadiCallbackSISO('f1',f ,in_rows=10,out_rows=5)
print(jax_model(np.transpose(a)))


#option['print_level']=0
opti_model = ca.Opti()
x = opti_model.variable(nx,horizon+1)


phi_1= x[0,:]
phi_2= x[1,:]
phi_3= x[2,:]
dphi_1= x[3,:]
dphi_2= x[4,:]
dphi_3= x[5,:]
phi_1_m= x[6,:]
phi_2_m= x[7,:]

phi_m_1_set = u[0,:]
phi_m_2_set = u[1,:]

k1 = casadi_model(x[:,0:-1],u)
k2 = casadi_model(x[:,0:-1]+dt/2*k1,u)
k3 = casadi_model(x[:,0:-1]+dt/2*k2,u)
k4 = casadi_model(x[:,0:-1]+dt*k3,u)

opti_jax.minimize(ca.dot(phi_1,phi_1)+ca.dot(phi_2,phi_2)+ca.dot(phi_3,phi_3)
              +0.001*ca.dot(phi_m_1_set,phi_m_1_set)+0.1*ca.dot(phi_m_2_set,phi_m_2_set)
              +0.001*ca.dot(phi_1_m,phi_1_m)+0.1*ca.dot(phi_2_m,phi_2_m))

opti_jax.subject_to(x[:,1:] == x[:,0:-1]+dt/6*(k1+2*k2+2*k3+k4))

opti_jax.subject_to(x[:,0]==x0)

opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,x[0:2,:],2*ca.pi))
opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,x[6:,:],2*ca.pi))
opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,u,2*ca.pi))


opti_jax.solver("ipopt",{},option)

try:
    jax_sol = opti_jax.solve()
    jax_phi_1 = opti_jax.value(phi_1)
    jax_phi_2 = opti_jax.value(phi_2)
    jax_phi_3 = opti_jax.value(phi_3)

except Exception as e:
    print(e)






# opts = {}
# opts["out_dim"] = [1, 1]
# opts["in_dim"] = [nd, 1]
opts={'in_dim':nd,'out_dim':1}
f = jax2tf.call_tf(lambda x: model.predict_y(tf.transpose(x))[0])

gpr = JaxCasadiCallbackSISO('f1',f ,in_rows=nd,out_rows=1)
print(gpr(arg))
#gpr = GPR(model, opts=opts)
arg1 = np.random.random((nd, 1))
print(gpr(arg1))


w = vertcat(*w)

# # Create an NLP solver
prob = {'f': gpr(w[0::3]), 'x': w, 'g': vertcat(*g)}
# options = {"ipopt": {"hessian_approximation": "limited-memory"}}
options = {"ipopt": {"hessian_approximation": "limited-memory"}}
solver = nlpsol('solver', 'ipopt', prob, options);
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

#print("Ncalls", gpr.counter)
#print("Total time [s]", gpr.time)
w_opt = sol['x'].full().flatten()

# Plot the solution
x1_opt = w_opt[0::3]
x2_opt = w_opt[1::3]
u_opt = w_opt[2::3]

print('total eval time:',gpr.time)
print(len(gpr.ref))
print(gpr.ref[0].time)

tgrid = [T / N * k for k in range(N + 1)]
import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
plt.xlabel('t')
plt.legend(['x1', 'x2', 'u'])
plt.grid()
plt.show()