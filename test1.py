from jax_casadi_callback import JaxCasadiCallbackSISO,JaxCasadiCallbackJacobian
import numpy as np
import casadi as ca
import jax
import jax.numpy as jnp
import timeit
import time

print(jax.devices())



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
option["hessian_approximation"] = "limited-memory"

casadi_option={'print_time':True,'enable_reverse': True}

def casadi_model(x):
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


ca.GlobalOptions.setMaxNumDir(1024*16)

N=20
#x = ca.MX.sym('x',8,N)
#u = ca.MX.sym('u',2,N)
y = ca.MX.sym('y',10,N)
dm_y = ca.DM_rand(10,N)

#print(dm_y.__array_custom__())



callback_f = JaxCasadiCallbackJacobian('jax_model_jac',jax.jit(lambda x:jax_full_model(x,N)),nx+nu,nx,N)
callback_jac = ca.Function('callback_jac',[y],[ca.jacobian(callback_f(y),y)])
callback_hes = ca.Function('callback_hes',[y],[ca.jacobian(callback_jac(y),y)])
callback_v = callback_f(dm_y)
callback_jac_v = callback_jac(dm_y)
callback_hes_v = callback_hes(dm_y)

casadi_f = ca.Function('casadi_f',[y],[casadi_model(y)])
casadi_jac = ca.Function('casadi_jac',[y],[ca.jacobian(casadi_model(y),y)])
casadi_hes = ca.Function('casadi_hes',[y],[ca.jacobian(casadi_jac(y),y)])



callback_v = callback_f(dm_y)
casadi_v = casadi_f(dm_y)

sum_test = (callback_v-casadi_v)**2
print('compare casadi and callback function: ',ca.sum2(ca.sum1(sum_test)))

casadi_jac_v = casadi_jac(dm_y)

sum_test = (casadi_jac_v-callback_jac_v)**2
print('compare casadi and callback jacobian: ',ca.sum2(ca.sum1(sum_test)))


print('callback jac cost time1',callback_f.jac_callback.time1)
print('callback jac cost time2',callback_f.jac_callback.time2)
print('callback jac cost time3',callback_f.jac_callback.time3)
print('callback jac its',callback_f.jac_callback.its)
print('callback function its',callback_f.its)
print()


dm_y[0,1]=0.5
callback_v = callback_f(dm_y)
casadi_v = casadi_f(dm_y)
sum_test = (callback_v-casadi_v)**2
print('compare casadi and callback function after modify input: ',ca.sum2(ca.sum1(sum_test)))

casadi_jac_v = casadi_jac(dm_y)
callback_jac_v = callback_jac(dm_y)
sum_test = (casadi_jac_v-callback_jac_v)**2
print('compare casadi and callback jacobian after modify input: ',ca.sum2(ca.sum1(sum_test)))


print('callback jac cost time1',callback_f.jac_callback.time1)
print('callback jac cost time2',callback_f.jac_callback.time2)
print('callback jac cost time3',callback_f.jac_callback.time3)
print('callback jac its',callback_f.jac_callback.its)
print('callback function its',callback_f.its)

print()
callback_v = callback_f(dm_y)
casadi_v = casadi_f(dm_y)
sum_test = (callback_v-casadi_v)**2
print('compare casadi and callback function second time but not modify input: ',ca.sum2(ca.sum1(sum_test)))

casadi_jac_v = casadi_jac(dm_y)
callback_jac_v = callback_jac(dm_y)
sum_test = (casadi_jac_v-callback_jac_v)**2
print('compare casadi and callback jacobian second time but not modify input: ',ca.sum2(ca.sum1(sum_test)))


print('callback jac cost time1',callback_f.jac_callback.time1)
print('callback jac cost time2',callback_f.jac_callback.time2)
print('callback jac cost time3',callback_f.jac_callback.time3)
print('callback jac its',callback_f.jac_callback.its)
print('callback function its',callback_f.its)


#print(jax_model_f(test_x0))
for i in range(100):
    callback_v = callback_f(dm_y)
    casadi_v = casadi_f(dm_y)
    casadi_jac_v = casadi_jac(dm_y)
    callback_jac_v = callback_jac(dm_y)



print('f time1',callback_f.time1)
print('f time2',callback_f.time2)
print('f time3',callback_f.time3)
print('f its',callback_f.its)

#print(len(callback_f.refs))
print('jac time1',callback_f.jac_callback.time1)
print('jac time2',callback_f.jac_callback.time2)
print('jac time3',callback_f.jac_callback.time3)
print('jac its',callback_f.jac_callback.its)

timer = timeit.Timer(lambda:callback_f(dm_y))
t = timer.timeit(1000)
print('call callback cost:',t)

timer = timeit.Timer(lambda:casadi_f(dm_y))
t = timer.timeit(1000)
print('call casadi cost:',t)

timer = timeit.Timer(lambda:callback_jac(dm_y))
t = timer.timeit(1000)
print('call callback jacobian cost:',t)

timer = timeit.Timer(lambda:casadi_jac(dm_y))
t = timer.timeit(1000)
print('call casadi jacobian cost:',t)

f = jax.jit(lambda x:jax_full_model(x,N))
t0 = time.time()
np_y = dm_y.full()

v = np.asarray(f(np_y))
print(time.time()-t0)




t = timeit.Timer(lambda:casadi_f(dm_x,dm_u))
print(t.timeit(1000))




#t = timeit.Timer(lambda:callback_siso_f(dm_y))
#print(t.timeit(1000))

#print(jax_model_f_siso.time)
#print(jax_model_f_siso.its)


t = timeit.Timer(lambda:callback_jac_f(dm_y))
print(t.timeit(1000))



print('time1',jax_model_f_jac.time1)
#t = timeit.Timer(lambda:dm_y.__float__())
#print(t.timeit(1000))
print('time2',jax_model_f_jac.time2)
print('time3',jax_model_f_jac.time3)
#print(jax_model_f_jac.its)





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