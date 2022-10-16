import casadi as ca
from numpy import dtype, float32, float64
import tensorflow as tf


#
# How to use Callback
# Joel Andersson
#

@tf.function
def f(x):
    #return tf.concat([tf.sin(x[2])*x[3]+x[3]**2, 2*x[0]+x[2], x[1]**2+5*x[2]],axis=0)
    #return tf.Variable([x[2], 2*x[0], x[1]])
    return tf.sin(x),tf.cos(x)

a=tf.constant([1, 2, 0, 3],dtype=float64)
print(f(a))


x = tf.Variable([1, 2, 0, 3],dtype=float64)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = f(x)
grad = tape.gradient(y,x)
grad1 = tape.gradient(y,x,output_gradients=tf.constant([1,0,0,0],dtype=float64))
grad2 = tape.gradient(y,x,output_gradients=tf.constant([0,0,0,0],dtype=float64))
print(y)
print(grad)
print(grad1)
print(grad2)

class MiMoCallback(ca.Callback):
    def __init__(self, name, opts={}):
        ca.Callback.__init__(self)        
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(4, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(3, 1)

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        #x, y = vertsplit(arg[0])
        #return [vertcat(y*exp(x), 3*x*y**2, 5*sin(x)+6*cos(y))]
        a, b, c, d = ca.vertsplit(arg[0])
        ret = ca.vertcat(ca.sin(c)*d+d**2, 2*a+c, b**2+5*c)
        return [ret]


class MiMoCallbackReverse(MiMoCallback):
    def has_reverse(self, nadj):
        return nadj == 1

    def get_reverse(self, nfwd, name, inames, onames, opts):
        # You are required to keep a reference alive to the returned Callback object
        class ReverseCallback(ca.Callback):
            def __init__(self, opts={}):
                ca.Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 3
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if (i == 1 or i == 2):  # nominal input
                    # print(i)
                    return ca.Sparsity.dense(3, 1)
                elif (i == 0):  # nominal output
                    return ca.Sparsity.dense(4, 1)

            def get_sparsity_out(self, i):
                # Reverse sensitivity
                return ca.Sparsity.dense(4, 1)

            # Evaluate numerically
            def eval(self, arg):
                print(arg)
                a,b,c,d = ca.vertsplit(arg[0])

                return [ca.DM([a, b,c,d])]
        self.rev_callback = ReverseCallback()
        return self.rev_callback


# f = MiMoCallbackReverse('f')
# x = MX.sym("x", 2)
# J = Function('J', [x], [jacobian(f(x), x)])
# print(J(vertcat(2, 1)))


f = MiMoCallbackReverse('g')
x = ca.MX.sym("x", 4)
J = ca.Function('J', [x], [ca.jacobian(f(x), x)])
print(J(ca.vertcat(1, 2, 0, 3)))