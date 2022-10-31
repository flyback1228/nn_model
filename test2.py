import numpy as np
import casadi as ca
import jax
import jax.numpy as jnp
import timeit
import time
from jax import config, jacfwd, jacrev
config.update('jax_enable_x64', True)

host_array = np.asarray(np.random.uniform(size=(800, 1000)),dtype=np.float64)
return_array = np.asarray(np.random.uniform(size=(800, 1000)),dtype=np.float64)

def f(x):
    return jnp.power(x,2)

t0 = time.time()
for i in range(100):
    jnp.asarray(host_array)
t1 = time.time()
print(t1-t0)

device_array = jnp.asarray(host_array)
device_array = device_array**2
print(type(device_array))

device_array = jnp.asarray(np.random.uniform(size=(800, 1000)),dtype=np.float64)

t0 = time.time()
for i in range(100):
    host_array[:,:] = device_array.__array__()
t1 = time.time()
print(t1-t0)

k = f(device_array)
print(k.device())

t0 = time.time()
print(host_array.shape)
host_array[:] = k
t1 = time.time()
print(t1-t0)