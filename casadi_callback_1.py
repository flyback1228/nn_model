import jax.numpy as jnp
import casadi as ca
import numpy as np
import jax

'''
'''
def concat_jac(f,rows):    
    return lambda *xs: jnp.hstack([jnp.reshape(jax.jacfwd(f,argnums=i)(*xs),(rows,-1)) for i in range(len(xs))])

class JaxCallback(ca.Callback):
    def __init__(self, name, f, opts={}):
        ca.Callback.__init__(self)
        self.f = f
        self.opts = opts
        self.construct(name, {})
        self.refs=[]

    # Number of inputs and outputs
    def get_n_in(self): return self.opts['n_in']
    def get_n_out(self): return len(self.f)
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.opts['in_dim'][i][0],self.opts['in_dim'][i][1])

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        jarg = [jnp.asarray(ar) for ar in arg]
        values = [self.f[i](*jarg) for i in range(self.n_out())]

        return_value = [np.asarray(jnp.reshape(self.f[i](*jarg),self.opts['out_dim'][i])) for i in range(self.n_out())]
        return return_value

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        #print(self.name_in)
        #print(self.name_out)
        print(name, inames, onames, opts)
        new_out=[]
        new_out_dim=[]
        opts={}
        cols = [self.opts['in_dim'][j][0]*self.opts['in_dim'][j][1] for j in range(self.opts['n_in'])]
        cols = np.sum(cols)
        for i,out_ in enumerate(self.f):
            new_out.append(concat_jac(out_,self.opts['out_dim'][i][0]*self.opts['out_dim'][i][1]))
            
            new_out_dim.append([self.opts['out_dim'][i][0]*self.opts['out_dim'][i][1],cols])
            #for j in range(self.opts['n_in']):
            #    new_out.append(jax.jacfwd(out_,argnums=j))
            #    new_out_dim.append([self.opts['out_dim'][i][0]*self.opts['out_dim'][i][1],self.opts['in_dim'][j][0]*self.opts['in_dim'][j][1]])

        opts['in_dim']=self.opts['in_dim']
        #opts['in_dim'].append(self.opts['out_dim'])
        opts['n_in']=self.opts['n_in']#+len(self.f)
        opts['out_dim']=new_out_dim
        callback = JaxCallback(name,new_out, opts=opts)

        value_x = ca.DM([1,2,3])
        value_y = ca.DM([3,1,2,1.5])
        y = callback(value_x,value_y)
        print(y)

        self.refs.append(callback)

        nominal_in = self.mx_in()        
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        casadi_bal = callback.call(nominal_in)
        #casadi_bal = [bal.T() for bal in casadi_bal]
        #out = ca.horzcat(*casadi_bal)
        return ca.Function(name, nominal_in + nominal_out, casadi_bal, inames, onames)
        


def f(x,y):
    return jnp.asarray([x[0]*y[0], 5*x[2]+y[1], 4*x[1]**2+y[1]*y[2] - 2*x[2], jnp.exp(y[3]) + x[2] * jnp.sin(x[0])])
opts={'in_dim':[[3,1],[4,1]],'out_dim':[[4,1]],'n_in':2}

evaluator = JaxCallback('f',[f],opts)

DM_x = ca.DM([1,2,3])
DM_y = ca.DM([3,1,2,1.5])
#print(f(jnp.asarray(value_x),jnp.asarray(value_y)))

jnp_x = jnp.asarray([1,2,3],dtype=jnp.float64)
jnp_y = jnp.asarray([3,1,2,1.5],dtype=jnp.float64)
z_jax = f(jnp_x,jnp_y)
print('jax evaluate:')
print(z_jax)

z_callback = evaluator(DM_x,DM_y)
print('callback evaluate:')
print(z_callback)

x = ca.MX.sym("x",3,1)
y = ca.MX.sym("x",4,1)
z = ca.vertcat(x[0]*y[0], 5*x[2]+y[1], 4*x[1]**2+y[1]*y[2] - 2*x[2], ca.exp(y[3]) + x[2] * ca.sin(x[0]))
F = ca.Function('F', [x,y], [z])
z_casadi = F(DM_x,DM_y)
print('casadi evaluate:')
print(z_casadi)
print('----------------------------------------')


jac_jax = jax.jacfwd(f)(jnp_x,jnp_y)
print('jax jacobian:')
print(jac_jax)

J_callback = ca.Function('J', [x,y], [ca.jacobian(evaluator(x,y), x)])
J_callback_y = ca.Function('J', [x,y], [ca.jacobian(evaluator(x,y), y)])
jac_callback = J_callback(DM_x,DM_y)
J_callback_y(DM_x,DM_y)
print('callback jacobian:')
print(jac_callback)

J_casadi = ca.Function('j_casadi', [x,y], [ca.jacobian(F(x,y), x)])
jac_casadi = J_casadi(DM_x,DM_y)
print('casasi jacobian:')
print(jac_casadi)
print('----------------------------------------')


hes_jax = jax.jacfwd(jax.jacfwd(f),argnums=0)(jnp_x,jnp_y)
print('jax hessian:')
print(hes_jax)

H_callback = ca.Function('H', [x,y], [ca.jacobian(J_callback(x,y), x)])
hes_callback = H_callback(DM_x,DM_y)
print('callback hessian:')
print(hes_callback)

#t = jnp.transpose(hes,axes=[2,0,1])
#print(t)

H_casadi = ca.Function('h_casadi', [x,y], [ca.jacobian(J_casadi(x,y), x)])
hes_casadi = H_casadi(DM_x,DM_y)
print('casasi hessian:')
print(hes_casadi)
print('----------------------------------------')

# H = ca.Function('h', [x,y], [ca.jacobian(J(x,y), y)])
# print(H(value_x,value_y))

# ca_y = ca.vertcat(x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * ca.sin(x[0]))
# ca_f = ca.Function('ca_f', [x], [ca_y])
# print(ca_f(value))
# ca_J = ca.Function('ca_J', [x], [ca.jacobian(ca_f(x), x)])
# ca_H = ca.Function('ca_h', [x], [ca.jacobian(ca_J(x), x)])
# print(ca_J(value))
# print(ca_H(value))
