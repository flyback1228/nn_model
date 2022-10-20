import jax.numpy as jnp
import casadi as ca
import numpy as np
import jax
from jax import config
config.update('jax_enable_x64', True)


'''
'''
def concat_jac(fs,in_dim):
    return [lambda *xs,i=i: jnp.hstack([jnp.reshape(jax.jacrev(f,argnums=i)(*xs).flatten('F'),(np.prod(in_dim[i]),-1)) for f in fs]) for i in range(len(in_dim))]
    

class ReverseCallback(ca.Callback):
    def __init__(self, name, f, opts={}):
        ca.Callback.__init__(self)
        self.f = f
        self.opts = opts
        self.arg_len = self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']        
        self.construct(name, {})
        self.extend_return = [ca.Sparsity(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1]) for i in range(len(self.f),self.n_out())]
        
    def get_n_in(self): return self.opts['n_in']
    def get_n_out(self): return self.opts['n_out']
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.opts['in_dim'][i][0],self.opts['in_dim'][i][1])

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1]) if i<len(self.f) else ca.Sparsity(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])

    def eval(self, arg):                
        jarg = [jnp.reshape(jnp.asarray(arg[i]),(np.prod(self.opts['in_dim'][i]),)) for i in range(self.arg_len)]
        return_value = [np.asarray(f(*jarg)) for f in self.f]
        return_value.extend(self.extend_return)
        return return_value

    def has_reverse(self,nadj):
        return True

    def get_reverse(self, nfwd, name, inames, onames, opts):       
        
        fs = concat_jac(self.f,self.opts['in_dim'][0:self.arg_len])

        new_opts={'n_in':self.opts['n_in']+2*self.opts['n_out'],'n_out':self.opts['n_in']}
        new_opts['original_n_in']=self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']
        in_dim = []
        in_dim.extend(self.opts['in_dim'])
        in_dim.extend(self.opts['out_dim'])
        in_dim.extend([[dim[0],dim[1]*nfwd] for dim in self.opts['out_dim']])        
        new_opts['in_dim'] = in_dim
        new_opts['out_dim'] = [[dim[0],dim[1]*nfwd] for dim in self.opts['in_dim']]

        self.rev_callback = ReverseCallback(name,fs,new_opts)
        return self.rev_callback


class JaxCallback(ca.Callback):
    def __init__(self, name, f, opts={}):
        ca.Callback.__init__(self)
        self.f = f
        self.opts = opts
        self.construct(name, {})

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
        jarg = [jnp.reshape(jnp.asarray(arg[i]),(np.prod(self.opts['in_dim'][i]),)) for i in range(len(arg))]
        #jarg = [jnp.reshape(jnp.asarray(arg[i]),(np.prod(self.opts['in_dim'][i]),)) for i in range(self.arg_len)]
        #values = [self.f[i](*jarg) for i in range(self.n_out())]

        return_value = [np.asarray(jnp.reshape(self.f[i](*jarg),self.opts['out_dim'][i])) for i in range(self.n_out())]
        
        return return_value

    def has_reverse(self,nadj):
        return True

    def get_reverse(self, nfwd, name, inames, onames, opts):        
        arg_len = self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']
        fs = concat_jac(self.f,self.opts['in_dim'][0:arg_len])
        in_dim = self.opts['in_dim']

        new_opts={'n_in':self.opts['n_in']+2*self.opts['n_out'],'original_n_in':self.opts['n_in'],'n_out':self.opts['n_in']}
        
        for dim in self.opts['out_dim']:
            in_dim.append(dim )
        for dim in self.opts['out_dim']:
            in_dim.append([dim[0],dim[1]*nfwd] )        
        new_opts['in_dim'] = in_dim

        out_dim=[]
        for i in range(arg_len):#self.opts['in_dim']:
            out_dim.append([in_dim[i][0],in_dim[i][1]*nfwd])
        new_opts['out_dim'] = out_dim

        self.rev_callback = ReverseCallback(name,fs,new_opts)
        return self.rev_callback


        


def f(x,y):
    return jnp.asarray([x[0]*y[0], 5*x[2]+y[1], 4*x[1]**2+y[1]*y[2] - 2*x[2], jnp.exp(y[3]) + x[2] * jnp.sin(x[0]),x[1]*y[2]])
opts={'in_dim':[[3,1],[4,1]],'out_dim':[[5,1]],'n_in':2,'n_out':1}

evaluator = ReverseCallback('f',[f],opts)

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
z = ca.vertcat(x[0]*y[0], 5*x[2]+y[1], 4*x[1]**2+y[1]*y[2] - 2*x[2], ca.exp(y[3]) + x[2] * ca.sin(x[0]),x[1]*y[2])
F = ca.Function('F', [x,y], [z])
z_casadi = F(DM_x,DM_y)
print('casadi evaluate:')
print(z_casadi)
print('----------------------------------------')


jac_jax = jax.jacfwd(f)(jnp_x,jnp_y)
print('jax jacobian:')
print(jac_jax)

J_callback = ca.Function('J', [x,y], [ca.jacobian(evaluator(x,y), y)])
#J_callback_y = ca.Function('J', [x,y], [ca.jacobian(evaluator(x,y), y)])
jac_callback = J_callback(DM_x,DM_y)
#J_callback_y(DM_x,DM_y)
print('callback jacobian:')
print(jac_callback)

J_casadi = ca.Function('j_casadi', [x,y], [ca.jacobian(F(x,y), y)])
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
