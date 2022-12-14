from casadi import *
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
from jax_casadi_callback import JaxCasadiCallback,JaxCasadiCallbackSISO


T = 10.  # Time horizon
N = 20  # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1, x2)
u = MX.sym('u')

# Model equations
xdot = vertcat((1 - x2 ** 2) * x1 - x2 + u, x1)

# Formulate discrete time dynamics
if False:
    # CVODES from the SUNDIALS suite
    dae = {'x': x, 'p': u, 'ode': xdot}
    opts = {'tf': T / N}
    F = integrator('F', 'cvodes', dae, opts)
else:
    # Fixed step Runge-Kutta 4 integrator
    M = 4  # RK4 steps per interval
    DT = T / N / M
    f = Function('f', [x, u], [xdot])
    X0 = MX.sym('X0', 2)
    U = MX.sym('U')
    X = X0
    Q = 0
    for j in range(M):
        k1 = f(X, U)
        k2 = f(X + DT / 2 * k1, U)
        k3 = f(X + DT / 2 * k2, U)
        k4 = f(X + DT * k3, U)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F = Function('F', [X0, U], [X], ['x0', 'p'], ['xf'])

# Start with an empty NLP
w = []
w0 = []
lbw = []
ubw = []
g = []
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += [0, 1]
ubw += [0, 1]
w0 += [0, 1]

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['xf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k + 1), 2)
    w += [Xk]
    lbw += [-0.25, -inf]
    ubw += [inf, inf]
    w0 += [0, 0]

    # Add equality constraint
    g += [Xk_end - Xk]
    lbg += [0, 0]
    ubg += [0, 0]

nd = N + 1

import gpflow
import time
import tensorflow as tf
from tensorflow_v2_casadi import TensorFlowEvaluator


class GPR(TensorFlowEvaluator):
    def __init__(self, model, opts={}):
        initializer = tf.random_normal_initializer(mean=1., stddev=2.)
        X = tf.Variable(initializer(shape=[1, nd], dtype=tf.float64), trainable=False)

        @tf.function
        def f_k(input_dat, get_grad_val=None):
            xf_tensor = input_dat
            if (get_grad_val is not None):
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(xf_tensor)
                    mean, _ = model.predict_y(xf_tensor)
                grad_mean = tape.gradient(mean, xf_tensor, output_gradients=get_grad_val)
            else:
                mean, _ = model.predict_y(xf_tensor)
                grad_mean = None
            return mean, grad_mean

        TensorFlowEvaluator.__init__(self, [X], [f_k], model, opts=opts)
        self.counter = 0
        self.time = 0

    def eval(self, arg):
        self.counter += 1
        t0 = time.time()
        ret = TensorFlowEvaluator.eval(self, arg)
        self.time += time.time() - t0
        return [ret]


# Create
np.random.seed(0)
data = np.random.normal(loc=0.5, scale=1, size=(1, nd))
value = np.random.random((1, 1))

model = gpflow.models.GPR(data=(data, value),
                          kernel=gpflow.kernels.Constant(nd) + gpflow.kernels.Linear(nd) + gpflow.kernels.White(
                              nd) + gpflow.kernels.RBF(nd))
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(
    model.training_loss,
    variables=model.trainable_variables,
    options=dict(disp=True, maxiter=100),
)
arg = np.random.random((nd, N))

pr = model.predict_y(np.transpose(arg))
print('model.predict_y',pr[0])

arg1 = np.random.random((nd, 1))
pr = model.predict_y(np.transpose(arg1))
print('model.predict_y',pr[0])

#import tensorflow as tf

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