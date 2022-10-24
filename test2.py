import numpy as np
import casadi as ca
import jax
import jax.numpy as jnp
from jax_casadi_callback import JaxCasadiCallbackSISO,JaxCasadiCallback


f_test = lambda x,y: jnp.asarray([x[0,:]*y[0,:], 5*x[2,:]+y[1,:], 4*x[1,:]**2+y[1,:]*y[2,:] - 2*x[2,:], jnp.exp(y[3,:]) + x[2,:] * jnp.sin(x[0,:]),x[1]*y[2,:]])

rows=1000
opts={'in_dim':[[3,rows],[4,rows]],'out_dim':[[5,rows]],'n_in':2,'n_out':1}
f_callback = JaxCasadiCallback('f1',f_test,opts)
#DM_x = ca.DM([1,2,3.0])
#DM_y = ca.DM([3,1,2,1.5])
DM_x = ca.DM_rand(3,rows)
DM_y = ca.DM_rand(4,rows)
jnp_x = jnp.asarray(DM_x,dtype=jnp.float32)
jnp_y = jnp.asarray(DM_y,dtype=jnp.float32)

v_callback = f_callback(DM_x,DM_y)
print('callback vjp evaluate:')
print(v_callback)
f_jax = jax.jit(f_test)

x = ca.MX.sym("x",3,rows)
y = ca.MX.sym("x",4,rows)
z = ca.vertcat(x[0,:]*y[0,:], 5*x[2,:]+y[1,:], 4*x[1,:]**2+y[1,:]*y[2,:] - 2*x[2,:], ca.exp(y[3,:]) + x[2,:] * ca.sin(x[0,:]),x[1,:]*y[2,:])
f_casadi = ca.Function('F', [x,y], [z])
v_casadi = f_casadi(DM_x,DM_y)
print('casadi original evaluate:')
print(v_casadi)


J_callback = ca.Function('J1', [x,y], [ca.jacobian(f_callback(x,y), x)])
v_jac_callback = J_callback(DM_x,DM_y)
print('callback vjp jacobian:')
print(v_jac_callback)
J_casadi = ca.Function('j_casadi', [x,y], [ca.jacobian(f_casadi(x,y), x)])
jac_casadi = J_casadi(DM_x,DM_y)
print('casasi jacobian:')
print(jac_casadi)

j_jax = jax.jit(jax.jacobian(f_jax))

H_callback = ca.Function('H1', [x,y], [ca.jacobian(J_callback(x,y), x)])
v_hes_callback = H_callback(DM_x,DM_y)
print('callback vjp hessian:')
print(v_hes_callback)
H_casadi = ca.Function('h_casadi', [x,y], [ca.jacobian(J_casadi(x,y), x)])
hes_casadi = H_casadi(DM_x,DM_y)
print('casasi hessian:')
print(hes_casadi)

h_jax = jax.jit(f_jax)
