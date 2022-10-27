import numpy as np
import casadi as ca
import jax
import jax.numpy as jnp
from jax_casadi_callback import JaxCasadiCallbackSISO,JaxCasadiCallback,JaxCasadiCallbackJacobian



theta_1 = theta_2 = theta_3 = 2.25e-4
c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
d = np.array([6.78,  8.01,  8.82])*1e-5
tau = 1e-2

horizon = 50
nx = 8
nu = 2
dt = 0.05
x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)

option = {}
option['max_iter']=10000


def jax_model(x):
    x=np.reshape(x,(-1,))
    phi_1= x[0]
    phi_2= x[1]
    phi_3= x[2]
    dphi_1= x[3]
    dphi_2= x[4]
    dphi_3= x[5]
    phi_1_m= x[6]
    phi_2_m= x[7]

    phi_m_1_set = x[8]
    phi_m_2_set = x[9]

    
    return jnp.asarray([
        dphi_1,
        dphi_2,
        dphi_3,
        -c[0]/theta_1*(phi_1-phi_1_m)-c[1]/theta_1*(phi_1-phi_2)-d[0]/theta_1*dphi_1,
        -c[1]/theta_2*(phi_2-phi_1)-c[2]/theta_2*(phi_2-phi_3)-d[1]/theta_2*dphi_2,
        -c[2]/theta_3*(phi_3-phi_2)-c[3]/theta_3*(phi_3-phi_2_m)-d[2]/theta_3*dphi_3,
        1/tau*(phi_m_1_set - phi_1_m),
        1/tau*(phi_m_2_set - phi_2_m)]
    ).reshape(-1,1)

def jax_full_model(x,N):    
    phi_1= x[0,:N]
    phi_2= x[1,:N]
    phi_3= x[2,:N]
    dphi_1= x[3,:N]
    dphi_2= x[4,:N]
    dphi_3= x[5,:N]
    phi_1_m= x[6,:N]
    phi_2_m= x[7,:N]

    phi_m_1_set = x[8,:N]
    phi_m_2_set = x[9,:N]

    
    return jnp.vstack([
        dphi_1,
        dphi_2,
        dphi_3,
        -c[0]/theta_1*(phi_1-phi_1_m)-c[1]/theta_1*(phi_1-phi_2)-d[0]/theta_1*dphi_1,
        -c[1]/theta_2*(phi_2-phi_1)-c[2]/theta_2*(phi_2-phi_3)-d[1]/theta_2*dphi_2,
        -c[2]/theta_3*(phi_3-phi_2)-c[3]/theta_3*(phi_3-phi_2_m)-d[2]/theta_3*dphi_3,
        1/tau*(phi_m_1_set - phi_1_m),
        1/tau*(phi_m_2_set - phi_2_m)]
    )

def casadi_model(x):
    #x=np.reshape(x,(-1,))
    phi_1= x[0]
    phi_2= x[1]
    phi_3= x[2]
    dphi_1= x[3]
    dphi_2= x[4]
    dphi_3= x[5]
    phi_1_m= x[6]
    phi_2_m= x[7]

    phi_m_1_set = x[8]
    phi_m_2_set = x[9]

    
    return ca.vertcat(
        dphi_1,
        dphi_2,
        dphi_3,
        -c[0]/theta_1*(phi_1-phi_1_m)-c[1]/theta_1*(phi_1-phi_2)-d[0]/theta_1*dphi_1,
        -c[1]/theta_2*(phi_2-phi_1)-c[2]/theta_2*(phi_2-phi_3)-d[1]/theta_2*dphi_2,
        -c[2]/theta_3*(phi_3-phi_2)-c[3]/theta_3*(phi_3-phi_2_m)-d[2]/theta_3*dphi_3,
        1/tau*(phi_m_1_set - phi_1_m),
        1/tau*(phi_m_2_set - phi_2_m)
    )

#vmap_model = jax.jit(jax.vmap(jax.jit(jax_model),in_axes=0,out_axes=0))
test_x = ca.MX.sym('x',10,1)
test_x0 = ca.vertcat(x0,0,0)

jax_model_f = JaxCasadiCallbackJacobian('jax_model',jax.jit(jax_model),nx+nu,nx)
ca_model_f = ca.Function('ca_model',[test_x],[casadi_model(test_x)])


print(test_x0)
print(test_x0.shape)
jac_jax_f = ca.Function('jac_jax_f',[test_x],[ca.jacobian(jax_model_f(test_x),test_x)])
jac_ca_f = ca.Function('jac_ca_f',[test_x],[ca.jacobian(ca_model_f(test_x),test_x)])

jac_jax = lambda x:jax.jacrev(jax_model)(x).reshape(8,-1)

hes_jax = jax.jacfwd(jac_jax)


print(jac_jax_f(test_x0))
print(jac_ca_f(test_x0))
print(jac_jax(jnp.array(test_x0)))
#print(jac_jax(jnp.array(test_x0)))

hes_jax_value = hes_jax(jnp.array(test_x0))
print(type(hes_jax_value))
print(len(hes_jax_value))
for v in hes_jax_value:
    print(v.shape)
hessian_f = ca.Function('f_hes',[test_x],[ca.jacobian(jac_jax_f(test_x),test_x)])
print(hessian_f(test_x0))



# opti_jax = ca.Opti()
# x = opti_jax.variable(nx+nu,horizon+1)
# #u = opti_jax.variable(nu,horizon)

# phi_1= x[0,:]
# phi_2= x[1,:]
# phi_3= x[2,:]
# dphi_1= x[3,:]
# dphi_2= x[4,:]
# dphi_3= x[5,:]
# phi_1_m= x[6,:]
# phi_2_m= x[7,:]

# phi_m_1_set = x[8,:]
# phi_m_2_set = x[9,:]

# # k1 = casadi_model(x[:,0:-1],u)
# # k2 = casadi_model(x[:,0:-1]+dt/2*k1,u)
# # k3 = casadi_model(x[:,0:-1]+dt/2*k2,u)
# # k4 = casadi_model(x[:,0:-1]+dt*k3,u)

# opti_jax.minimize(ca.dot(phi_1,phi_1)+ca.dot(phi_2,phi_2)+ca.dot(phi_3,phi_3)
#               +0.001*ca.dot(phi_m_1_set,phi_m_1_set)+0.1*ca.dot(phi_m_2_set,phi_m_2_set)
#               +0.001*ca.dot(phi_1_m,phi_1_m)+0.1*ca.dot(phi_2_m,phi_2_m))

# opti_jax.subject_to(x[0:8,1:] == x[0:8,0:-1]+dt*jax_model_f(x[:,0:-1]))

# opti_jax.subject_to(x[0:8,0]==x0)

# opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_1,2*ca.pi))
# opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_2,2*ca.pi))
# opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_3,2*ca.pi))

# opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_1_m,2*ca.pi))
# opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_1_m,2*ca.pi))

# opti_jax.subject_to(opti_jax.bounded(-ca.pi,phi_m_1_set,ca.pi))
# opti_jax.subject_to(opti_jax.bounded(-ca.pi,phi_m_2_set,ca.pi))

# opti_jax.solver("ipopt",{},option)

# try:
#     jax_sol = opti_jax.solve()
#     jax_phi_1 = opti_jax.value(phi_1)
#     jax_phi_2 = opti_jax.value(phi_2)
#     jax_phi_3 = opti_jax.value(phi_3)

# except Exception as e:
#     print(e)