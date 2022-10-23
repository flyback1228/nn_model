import casadi as ca
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

from casadi import Sparsity
import gpflow
import numpy as np


class TensorFlowEvaluator(ca.Callback):
    def __init__(self, t_in, t_out, model, set_init=False, opts={}):

        self.set_init = set_init
        self.opts = opts
        ca.Callback.__init__(self)
        assert isinstance(t_in, list)
        self.t_in = t_in
        assert isinstance(t_out, list)
        self.t_out = t_out
        self.output_shapes = []
        self.construct("TensorFlowEvaluator", {})
        self.refs = []
        self.model = model

    def get_n_in(self):
        return len(self.t_in)

    def get_n_out(self):
        return len(self.t_out)

    def get_sparsity_in(self, i):
        tesnor_shape = self.t_in[i].shape
        return Sparsity.dense(tesnor_shape[0], tesnor_shape[1])

    def get_sparsity_out(self, i):
        if (i == 0 and self.set_init is False):
            tensor_shape = [self.opts["output_dim"][0], self.opts["output_dim"][1]]
        elif (i == 0 and self.set_init is True):
            tensor_shape = [self.opts["grad_dim"][0], self.opts["grad_dim"][1]]
        else:
            tensor_shape = [self.opts["output_dim"][0], self.opts["output_dim"][1]]
        return Sparsity.dense(tensor_shape[0], tensor_shape[1])

    def eval(self, arg):
        updated_t = []
        for i, v in enumerate(self.t_in):
            updated_t.append(tf.Variable(arg[i].toarray()))
        if (len(updated_t) == 1):
            out_, grad_estimate = self.t_out[0](tf.convert_to_tensor(updated_t[0]))
            selected_set = out_.numpy()
        else:
            out_, grad_estimate = self.t_out[0](tf.convert_to_tensor(updated_t[0]), tf.convert_to_tensor(updated_t[1]))
            selected_set = grad_estimate.numpy()

        return [selected_set]

    # Vanilla tensorflow offers just the reverse mode AD
    def has_reverse(self, nadj):
        return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        initializer = tf.random_normal_initializer(mean=1., stddev=2.)
        adj_seed = [tf.Variable(initializer(shape=self.sparsity_out(i).shape, dtype=tf.float64)) for i in
                    range(self.n_out())]

        callback = TensorFlowEvaluator(self.t_in + adj_seed, [self.t_out[0]], self.model, set_init=True, opts=self.opts)
        self.refs.append(callback)

        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        casadi_bal = callback.call(nominal_in + adj_seed)
        return ca.Function(name, nominal_in + nominal_out + adj_seed, casadi_bal, inames, onames)
