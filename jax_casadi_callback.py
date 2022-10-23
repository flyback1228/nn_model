import jax.numpy as jnp
import casadi as ca
import numpy as np
import jax
from jax import config
config.update('jax_enable_x64', True)
import copy

class JaxCasadiCallback(ca.Callback):
    def __init__(self, name, f, opts={}):
        ca.Callback.__init__(self)
        self.f = jax.jit(f)
        
        self.opts = opts
        self.arg_len = self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']       
        rev_f = lambda x: jax.vmap(jax.vjp(self.f,*x[0:self.arg_len])[1],in_axes=1,out_axes=-1)(*x[self.opts['n_in']+self.opts['n_out']:self.opts['n_in']+2*self.opts['n_out']]) 
        self.rev_f = jax.jit(rev_f)
        self.construct(name, {})
        
        
    def get_n_in(self): return self.opts['n_in']
    def get_n_out(self): return self.opts['n_out']
    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.opts['in_dim'][i][0],self.opts['in_dim'][i][1])

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])# if i<len(self.f) else ca.Sparsity(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])

    def eval(self, arg):                
        jarg = [jnp.asarray(arg[i]) for i in range(self.arg_len)]
        return_value = np.asarray(self.f(*jarg)) 
        return [return_value]

    def has_reverse(self,nadj):
        return True

    def get_reverse(self, nfwd, name, inames, onames, opts):       
        
        class ReverseFun(ca.Callback):
            def __init__(self,name,f,opts):
                ca.Callback.__init__(self)
                self.f = f
                self.opts=opts
                rev_f = lambda x: jax.vmap(jax.vjp(self.f,x[0:self.opts['n_in']])[1],in_axes=1,out_axes=-1)(tuple(x[self.opts['n_in']+self.opts['n_out']:self.opts['n_in']+2*self.opts['n_out']])) 
                self.rev_f = jax.jit(rev_f)
                self.construct(name, {})

            def get_n_in(self): return self.opts['n_in']
            def get_n_out(self): return self.opts['n_out']

            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(self.opts['in_dim'][i][0],np.prod(self.opts['in_dim'][i][1:]))

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:]))

            # Evaluate numerically
            def eval(self, arg): 
                jarg = [jnp.squeeze(jnp.reshape(jnp.asarray(arg[i]),self.opts['in_dim'][i])) for i in range(len(arg)) ] 
                ret = self.f(jarg)
                ret=ret[0] if isinstance(ret[0], (list, tuple)) else ret 
                return [np.asarray(jnp.reshape(ret[i],(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:])))) for i in range(len(ret))]

            def has_reverse(self, nadj):
                return True
            
            def get_reverse(self, nfwd, name, inames, onames, opts): 
                new_opts={'n_in':self.opts['n_in']+2*self.opts['n_out'],'n_out':self.opts['n_in']}
                new_opts['original_n_in']=self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']
                in_dim = []
                in_dim.extend(copy.deepcopy(self.opts['in_dim']))
                in_dim.extend(copy.deepcopy(self.opts['out_dim']))
                in_dim.extend(copy.deepcopy(self.opts['out_dim']))
                for i in range(self.opts['n_out']):
                    in_dim[i+self.opts['n_in']+self.opts['n_out']].insert(1,nfwd)   
                new_opts['in_dim'] = in_dim
                new_opts['out_dim'] = [dim+[nfwd] for dim in self.opts['in_dim']]
                self.rev_callback = ReverseFun(name,self.rev_f,new_opts)
                return self.rev_callback



        
        new_opts={'n_in':self.opts['n_in']+2*self.opts['n_out'],'n_out':self.opts['n_in'],'n_fwd':nfwd}
        new_opts['original_n_in']=self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']
        in_dim = []
        in_dim.extend(copy.deepcopy(self.opts['in_dim']))
        in_dim.extend(copy.deepcopy(self.opts['out_dim']))
        in_dim.extend(copy.deepcopy(self.opts['out_dim']))
        for i in range(self.opts['n_out']):
            in_dim[i+self.opts['n_in']+self.opts['n_out']].insert(1,nfwd)   
        new_opts['in_dim'] = in_dim
        new_opts['out_dim'] = [dim+[nfwd] for dim in self.opts['in_dim']]
        self.rev_callback = ReverseFun(name,self.rev_f,new_opts)
        return self.rev_callback



if __name__ =='__main__':
    f_test = lambda x,y: jnp.asarray([x[0]*y[0], 5*x[2]+y[1], 4*x[1]**2+y[1]*y[2] - 2*x[2], jnp.exp(y[3]) + x[2] * jnp.sin(x[0]),x[1]*y[2]])
    
    opts={'in_dim':[[3,1],[4,1]],'out_dim':[[5,1]],'n_in':2,'n_out':1}
    evaluator1 = JaxCasadiCallback('f1',f_test,opts)
    DM_x = ca.DM([1,2,3.0])
    DM_y = ca.DM([3,1,2,1.5])

    jnp_x = jnp.asarray([1.0,2,3],dtype=jnp.float32)
    jnp_y = jnp.asarray([3.0,1,2,1.5],dtype=jnp.float32)

    print('---------------------------------')
    print('--------------------------------')

    z_callback = evaluator1(DM_x,DM_y)
    print('callback vjp evaluate:')
    print(z_callback)

    x = ca.MX.sym("x",3,1)
    y = ca.MX.sym("x",4,1)
    z = ca.vertcat(x[0]*y[0], 5*x[2]+y[1], 4*x[1]**2+y[1]*y[2] - 2*x[2], ca.exp(y[3]) + x[2] * ca.sin(x[0]),x[1]*y[2])
    F = ca.Function('F', [x,y], [z])
    z_casadi = F(DM_x,DM_y)
    print('casadi original evaluate:')
    print(z_casadi)


    J_callback = ca.Function('J1', [x,y], [ca.jacobian(evaluator1(x,y), x)])
    jac_callback = J_callback(DM_x,DM_y)
    print('callback vjp jacobian:')
    print(jac_callback)
    J_casadi = ca.Function('j_casadi', [x,y], [ca.jacobian(F(x,y), x)])
    jac_casadi = J_casadi(DM_x,DM_y)
    print('casasi jacobian:')
    print(jac_casadi)

    H_callback = ca.Function('H1', [x,y], [ca.jacobian(J_callback(x,y), x)])
    hes_callback = H_callback(DM_x,DM_y)
    print('callback vjp hessian:')
    print(hes_callback)
    H_casadi = ca.Function('h_casadi', [x,y], [ca.jacobian(J_casadi(x,y), x)])
    hes_casadi = H_casadi(DM_x,DM_y)
    print('casasi hessian:')
    print(hes_casadi)

    print('---------------------------------')
    print('---------------------------------')