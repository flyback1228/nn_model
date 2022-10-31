from jax_casadi_callback import JaxCasadiCallbackJacobian
import numpy as np
import casadi as ca
import jax.numpy as jnp
import jax 

horizon = 50
nx = 8
nu = 2
dt = 0.05
x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)

theta_1 = theta_2 = theta_3 = 2.25e-4
c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
d = np.array([6.78,  8.01,  8.82])*1e-5
tau = 1e-2


option = {}
option['max_iter']=10000
option["hessian_approximation"] = "limited-memory"

casadi_option={'print_time':True,'enable_reverse': True}

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

#vmap_model = jax.vmap(jax_model,in_axes=0,out_axes=0)

def jax_full_model(x,N):  

    return jnp.asarray([
        x[3,:N],
        x[4,:N],
        x[5,:N],
        -c[0]/theta_1*(x[0,:N]-x[6,:N])-c[1]/theta_1*(x[0,:N]-x[1,:N])-d[0]/theta_1*x[3,:N],
        -c[1]/theta_2*(x[1,:N]-x[0,:N])-c[2]/theta_2*(x[1,:N]-x[2,:N])-d[1]/theta_2*x[4,:N],
        -c[2]/theta_3*(x[2,:N]-x[1,:N])-c[3]/theta_3*(x[2,:N]-x[7,:N])-d[2]/theta_3*x[5,:N],
        1/tau*(x[8,:N] - x[6,:N]),
        1/tau*(x[9,:N] - x[7,:N])],dtype=np.float64
    )

#jax_model_f = JaxCasadiCallbackJacobian('jax_model',jax_model,nx+nu,nx,horizon+1)
jax_model_f = JaxCasadiCallbackJacobian('jax_model_jac',jax.jit(lambda x:jax_full_model(x,horizon)),nx+nu,nx,horizon)


test_x = ca.MX.sym('test_x',nx+nu,horizon)
jac_f = ca.Function('f_jac',[test_x],[ca.jacobian(jax_model_f(test_x),test_x)])
dm_y = ca.DM_rand(nx+nu,horizon)

it1 = jax_model_f.callback.its
callback_jac_v = jac_f(dm_y)
it2 = jax_model_f.callback.its

print(it1)
print(it2)
jax_model_f.callback.total_N = it2-it1



#option['print_level']=0
opti_jax = ca.Opti()
x = opti_jax.variable(nx+nu,horizon+1)
#u = opti_jax.variable(nu,horizon)

phi_1= x[0,:]
phi_2= x[1,:]
phi_3= x[2,:]
dphi_1= x[3,:]
dphi_2= x[4,:]
dphi_3= x[5,:]
phi_1_m= x[6,:]
phi_2_m= x[7,:]

phi_m_1_set = x[8,:]
phi_m_2_set = x[9,:]

# k1 = casadi_model(x[:,0:-1],u)
# k2 = casadi_model(x[:,0:-1]+dt/2*k1,u)
# k3 = casadi_model(x[:,0:-1]+dt/2*k2,u)
# k4 = casadi_model(x[:,0:-1]+dt*k3,u)

opti_jax.minimize(ca.dot(phi_1,phi_1)+ca.dot(phi_2,phi_2)+ca.dot(phi_3,phi_3)
              +0.001*ca.dot(phi_m_1_set,phi_m_1_set)+0.1*ca.dot(phi_m_2_set,phi_m_2_set)
              +0.001*ca.dot(phi_1_m,phi_1_m)+0.1*ca.dot(phi_2_m,phi_2_m))

opti_jax.subject_to(x[0:8,1:] == x[0:8,0:-1]+dt*jax_model_f(x[:,0:-1]))

opti_jax.subject_to(x[0:8,0]==x0)

opti_jax.subject_to(x[8:,-1]==0)


opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_1,2*ca.pi))
opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_2,2*ca.pi))
opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_3,2*ca.pi))

opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_1_m,2*ca.pi))
opti_jax.subject_to(opti_jax.bounded(-2*ca.pi,phi_1_m,2*ca.pi))

opti_jax.subject_to(opti_jax.bounded(-ca.pi,phi_m_1_set,ca.pi))
opti_jax.subject_to(opti_jax.bounded(-ca.pi,phi_m_2_set,ca.pi))

opti_jax.solver("ipopt",{},option)

try:
    jax_sol = opti_jax.solve()
    jax_phi_1 = opti_jax.value(phi_1)
    jax_phi_2 = opti_jax.value(phi_2)
    jax_phi_3 = opti_jax.value(phi_3)

except Exception as e:
    print(e)

print(jax_model_f.callback.its)
print(jax_model_f.callback.total_N)
print(jax_model_f.callback.its/jax_model_f.callback.total_N)