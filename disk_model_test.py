from numpy import dtype
from tensorflow import float64
from disk_models import *
import tensorflow as tf




print(tf.__version__)

horizon = 20
nx = 8
nu = 2
dt = 0.05
x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1,1)
u0 = np.pi*np.array([-1,0.5]).reshape(-1,1)

tf_x0 = tf.constant(x0)
tf_u0 = tf.constant(u0)
adj_seed = tf.constant([1,0,0,0,0,0,0,0],dtype=float64)
with tf.GradientTape() as tape:
    
    tape.watch(tf_x0)    
    tf_x_dot = tf_model(tf_x0,tf_u0)
    tf_grad = tape.gradient(tf_x_dot,tf_x0,output_gradients=adj_seed)

    print(tf_grad)

#tf_grad = tf.reshape(tf_grad,-1)
#print(tf_grad.shape)
#print(tf.split(tf_grad,8))



