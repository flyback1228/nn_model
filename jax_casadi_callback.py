import jax.numpy as jnp
import casadi as ca
import numpy as np
import jax
from jax import config, jacfwd, jacrev
config.update('jax_enable_x64', True)
import copy
import time

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

class ReverseFun(ca.Callback):
    def __init__(self,name,f,opts):
        ca.Callback.__init__(self)
        self.f = jax.jit(f)
        self.opts=opts
        self.rev_f = jax.jit(lambda x: jax.vjp(self.f,x[0:self.opts['n_in']])[1](tuple(x[self.opts['n_in']+self.opts['n_out']:self.opts['n_in']+2*self.opts['n_out']])) )
        #self.rev_f = rev_f)
        self.construct(name, {'jit':True})
        self.time=0.0
        

    def get_n_in(self): return self.opts['n_in']
    def get_n_out(self): return self.opts['n_out']

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(self.opts['in_dim'][i][0],np.prod(self.opts['in_dim'][i][1:]))

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:]))

    # Evaluate numerically
    def eval(self, arg):
        #print('arg len:',len(arg))
        #jarg = [jnp.asarray(arg[i]) for i in range(len(arg))]
        #t0 = time.time()
        jarg = [ar.full() for ar in arg]
        ret = self.f(jarg)
        ret=ret[0] if isinstance(ret[0], (list, tuple)) else ret 
        #self.time += time.time()-t0
        #return [np.asarray(jnp.reshape(ret[i],(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:])))) for i in range(len(ret))]
        return [r.__array__() for r in ret]

    def has_reverse(self, nadj):
        #print(nadj)
        return nadj==1
    
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

class JaxCasadiCallbackSISO(ca.Callback):
    def __init__(self, name, f, in_rows,out_rows):
        ca.Callback.__init__(self)
        self.f = jax.jit(f)
        self.in_rows=in_rows
        self.out_rows=out_rows
        #self.opts = opts
        self.rev_f = jax.jit(lambda x: jax.vjp(self.f,x[0])[1](x[2]) )
        # = rev_f)
        self.construct(name, {"enable_fd": True})
        self.time = 0
        self.its = 0
        self.ref=[]
        
        
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        #return ca.Sparsity.dense(self.opts['in_dim'][i][0],self.opts['in_dim'][i][1])
        return ca.Sparsity.dense(self.in_rows,1)

    def get_sparsity_out(self, i):
        #return ca.Sparsity.dense(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])
        return ca.Sparsity.dense(self.out_rows,1)

    def eval(self, arg):
        #print(arg)
        #print(jnp.asarray(arg[0]))
        t0 = time.time()
        return_value = self.f(arg[0].full()).__array__()
        self.time += time.time()-t0
        self.its+=1
        return [return_value]

    def has_reverse(self,nadj):
        return nadj==1

    def get_reverse(self, nfwd, name, inames, onames, opts):       
        #print('nfwd,',nfwd) 
        new_opts={'n_in':3,'n_out':1,'n_fwd':nfwd}
        #new_opts['original_n_in']=self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']
        new_opts['in_dim'] = [[self.in_rows,1],[self.out_rows,1],[self.out_rows, 1]]
        #in_dim.extend(copy.deepcopy(self.opts['in_dim']))
        #in_dim.extend(copy.deepcopy(self.opts['out_dim']))
        #in_dim.extend(copy.deepcopy(self.opts['out_dim']))
        #for i in range(self.opts['n_out']):
        #    in_dim[i+self.opts['n_in']+self.opts['n_out']].insert(1,nfwd)
        #new_opts['in_dim'] = in_dim
        #new_opts['out_dim'] = [dim+[nfwd] for dim in self.opts['in_dim']]
        new_opts['out_dim'] = [[self.in_rows,1]]
        rev_callback = ReverseFun(name,self.rev_f,new_opts)

        nominal_in = self.mx_in()        
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        casadi_bal = rev_callback.call(nominal_in+ nominal_out+adj_seed)

        self.ref.append(rev_callback)
        #return rev_callback
        return ca.Function(name, nominal_in + nominal_out+adj_seed, casadi_bal, inames, onames)

class JaxCasadiCallbackMIMO(ca.Callback):
    def __init__(self, name, f, in_rows,out_rows,N):
        ca.Callback.__init__(self)
        self.f = jax.jit(jax.vmap(f,in_axes=-1,out_axes=-1))
        self.rev_f = jax.jit(lambda x: jax.vjp(self.f,x[0])[1](x[2]))
        self.in_rows=in_rows
        self.out_rows=out_rows
        self.N = N
        #self.opts = opts

        #self.rev_f = jax.vmap(lambda x: jax.vjp(f,x[0])[1](x[2]) ,in_axes=0,)
        # = rev_f)
        self.construct(name, {})
        self.time = 0
        self.ref=[]
        
        
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        #return ca.Sparsity.dense(self.opts['in_dim'][i][0],self.opts['in_dim'][i][1])
        return ca.Sparsity.dense(self.in_rows,self.N)

    def get_sparsity_out(self, i):
        #return ca.Sparsity.dense(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])
        return ca.Sparsity.dense(self.out_rows,self.N)

    def eval(self, arg):
        #print(arg)
        print(jnp.asarray(arg[0]))
        #t0 = time.time()
        return_value = self.f(arg[0].full()).__array__()
        #self.time += time.time()-t0
        return [return_value]

    def has_reverse(self,nadj):
        return nadj==1

    def get_reverse(self, nfwd, name, inames, onames, opts):       
        class ReverseFunMIMO(ca.Callback):
            def __init__(self,name,f,opts):
                ca.Callback.__init__(self)
                self.f = f
                self.opts=opts
                self.rev_f = jax.jit(lambda x: jax.vjp(self.f,x[0:self.opts['n_in']])[1](tuple(x[self.opts['n_in']+self.opts['n_out']:self.opts['n_in']+2*self.opts['n_out']])) )
                #self.rev_f = rev_f)
                self.construct(name, {})
                self.time=0
                self.ref=[]
                

            def get_n_in(self): return self.opts['n_in']
            def get_n_out(self): return self.opts['n_out']

            def get_sparsity_in(self, i):
                return ca.Sparsity.dense(self.opts['in_dim'][i][0],np.prod(self.opts['in_dim'][i][1:]))

            def get_sparsity_out(self, i):
                return ca.Sparsity.dense(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:]))

            # Evaluate numerically
            def eval(self, arg):
                #print('arg len:',len(arg))
                #jarg = [jnp.asarray(arg[i]) for i in range(len(arg))]
                #t0 = time.time()
                self.time+=1
                jarg = [ar.full() for ar in arg]
                #print('jarg',jarg)
                ret = self.f(jarg)
                ret=ret[0] if isinstance(ret[0], (list, tuple)) else ret 
                #self.time += time.time()-t0
                #return [np.asarray(jnp.reshape(ret[i],(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:])))) for i in range(len(ret))]
                return [r.__array__() for r in ret]

            def has_reverse(self, nadj):
                return nadj==1
            
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
                rev_callback = ReverseFun(name,self.rev_f,new_opts)
                self.ref.append[self.rev_callback]
                nominal_in = self.mx_in()
                nominal_out = self.mx_out()
                adj_seed = self.mx_out()
                return ca.Function(name,nominal_in+nominal_out+adj_seed,rev_callback.call(nominal_in+adj_seed),inames,onames)
                
                #return rev_callback 
        new_opts={'n_in':3,'n_out':1,'n_fwd':nfwd}
        #new_opts['original_n_in']=self.opts['n_in'] if not 'original_n_in' in self.opts else self.opts['original_n_in']
        new_opts['in_dim'] = [[self.in_rows,self.N],[self.out_rows,self.N],[self.out_rows, self.N*nfwd]]
        #in_dim.extend(copy.deepcopy(self.opts['in_dim']))
        #in_dim.extend(copy.deepcopy(self.opts['out_dim']))
        #in_dim.extend(copy.deepcopy(self.opts['out_dim']))
        #for i in range(self.opts['n_out']):
        #    in_dim[i+self.opts['n_in']+self.opts['n_out']].insert(1,nfwd)
        #new_opts['in_dim'] = in_dim
        #new_opts['out_dim'] = [dim+[nfwd] for dim in self.opts['in_dim']]
        new_opts['out_dim'] = [[self.in_rows,self.N]]
        rev_callback = ReverseFunMIMO(name,self.rev_f,new_opts)
        self.ref.append(rev_callback)
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return ca.Function(name,nominal_in+nominal_out+adj_seed,rev_callback.call(nominal_in+nominal_out+adj_seed),inames,onames)
        #return rev_callback

def concat_jac(f,rows):    
    return lambda *xs: jnp.hstack([jnp.reshape(jax.jacfwd(f,argnums=i)(*xs),(rows,-1)) for i in range(len(xs))])


class ReverseFunJacbian(ca.Callback):
    def __init__(self,name,f,opts):
        ca.Callback.__init__(self)
        #self.f = jax.jit(f)
        self.f = jax.jit(lambda x:jacrev(f)(x).reshape(*self.opts['out_dim'][0]))
        self.opts=opts
        #self.rev_f = jax.jit(lambda x: jax.vjp(self.f,x[0:self.opts['n_in']])[1](tuple(x[self.opts['n_in']+self.opts['n_out']:self.opts['n_in']+2*self.opts['n_out']])) )
        #self.rev_f = rev_f)
        self.construct(name, {})
        self.time=0.0
        

    def get_n_in(self): return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(*self.opts['in_dim'][i])

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(*self.opts['out_dim'][i])

    def has_eval_buffer(self):
        return True

    def eval_buffer(self, arg, res) -> "int":
        a = np.frombuffer(arg[0], dtype=np.float64).reshape(self.opts['in_dim'][0],order='F')
        return_value = self.f(a)
        r0 = np.frombuffer(res[0], dtype=np.float64).reshape((self.opts['out_dim'][0]),order='F')
        r0[:] = np.squeeze(np.asarray(return_value))
        #ret=ret[0] if isinstance(ret[0], (list, tuple)) else ret 
        #self.time += time.time()-t0
        #return [np.asarray(jnp.reshape(ret[i],(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:])))) for i in range(len(ret))]                 
        return 0

        # t0 = time.time()
        # a = np.frombuffer(arg[0], dtype=np.float64).reshape((self.in_rows,-1),order='F')
        # #a = np.ndarray(arg[0],dtype=np.float64,copy=False,subok=True)#.__array_custom__()
        # #print(a)
        # t1 = time.time()
        # self.time1 += t1-t0
        # return_value = self.f(a)#.reshape(self.out_rows,self.N).__array__()
        # t2 = time.time()
        # self.time2 += t2-t1
        # r0 = np.frombuffer(res[0], dtype=np.float64).reshape((self.out_rows,self.N),order='F')
        # r0[:] = np.squeeze(np.asarray(return_value))#.__array__()#.reshape(self.out_rows,self.N).__array__()
        # t3 = time.time()
        # self.time3 += t3-t2
        # return 0

    # Evaluate numerically
    # def eval(self, arg):
    #     #print('arg len:',len(arg))
    #     #jarg = [jnp.asarray(arg[i]) for i in range(len(arg))]
    #     #t0 = time.time()
    #     jarg = [ar.full() for ar in arg]
    #     ret = self.f(jarg[0]).__array__()
    #     #ret=ret[0] if isinstance(ret[0], (list, tuple)) else ret 
    #     #self.time += time.time()-t0
    #     #return [np.asarray(jnp.reshape(ret[i],(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:])))) for i in range(len(ret))]                 
    #     return [ret]

    # def has_jacobian(self, *args) -> "bool":
    #     return True
    # def get_jacobian(self, name, inames, onames,opts):
    #     class ReverseFunHessian(ca.Callback):
    #         def __init__(self,name,f,opts):
    #             ca.Callback.__init__(self)
    #             #self.f = jax.jit(f)
    #             self.f = jax.jit(lambda x:jacrev(f)(x).reshape(*self.opts['out_dim'][0]))
    #             self.opts=opts
    #             #self.rev_f = jax.jit(lambda x: jax.vjp(self.f,x[0:self.opts['n_in']])[1](tuple(x[self.opts['n_in']+self.opts['n_out']:self.opts['n_in']+2*self.opts['n_out']])) )
    #             #self.rev_f = rev_f)
    #             self.construct(name, {})
    #             self.time=0.0
                

    #         def get_n_in(self): return 2
    #         def get_n_out(self): return 1

    #         def get_sparsity_in(self, i):
    #             return ca.Sparsity.dense(*self.opts['in_dim'][i])

    #         def get_sparsity_out(self, i):
    #             return ca.Sparsity.dense(*self.opts['out_dim'][i])

    #         # Evaluate numerically
    #         def eval(self, arg):
    #             #print('arg len:',len(arg))
    #             #jarg = [jnp.asarray(arg[i]) for i in range(len(arg))]
    #             #t0 = time.time()
    #             jarg = [ar.full() for ar in arg]
    #             ret = self.f(jarg[0]).__array__()
    #             #ret=ret[0] if isinstance(ret[0], (list, tuple)) else ret 
    #             #self.time += time.time()-t0
    #             #return [np.asarray(jnp.reshape(ret[i],(self.opts['out_dim'][i][0],np.prod(self.opts['out_dim'][i][1:])))) for i in range(len(ret))]                 
    #             return [ret]

    #     new_opts={}
    #     new_opts['in_dim'] = [[self.in_rows,1],[self.out_rows,1]]
    #     new_opts['out_dim'] = [[self.out_rows,self.in_rows]] 
    #     callback = ReverseFunHessian(name,f = self.f, opts=new_opts)
    #     self.refs.append(callback)
    #     nominal_in = self.mx_in()        
    #     nominal_out = self.mx_out()
    #     #adj_seed = self.mx_out()
    #     casadi_bal = callback.call(nominal_in)
    #     #casadi_bal = [bal.T() for bal in casadi_bal]
    #     #out = ca.horzcat(*casadi_bal)
    #     return ca.Function(name, nominal_in + nominal_out, casadi_bal, inames, onames)
        


class JaxCasadiCallbackJacobian(ca.Callback):
    def __init__(self, name, f, in_rows,out_rows,N):
        ca.Callback.__init__(self)
        #self.f = jax.jit(jax.vmap(jax.jit(f),in_axes=-1,out_axes=-1))
        self.f = f
        self.in_rows=in_rows
        self.out_rows=out_rows        
        self.refs=[]
        self.N = N
        self.time1=0
        self.time2=0
        self.time3=0
        self.its=0

        self.construct(name, {})

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i):
        #return ca.Sparsity.dense(self.opts['in_dim'][i][0],self.opts['in_dim'][i][1])
        return ca.Sparsity.dense(self.in_rows,self.N)

    def get_sparsity_out(self, i):
        #return ca.Sparsity.dense(self.opts['out_dim'][i][0],self.opts['out_dim'][i][1])
        return ca.Sparsity.dense(self.out_rows,self.N)

    # Initialize the object
    def init(self):
        print('initializing object')

    def has_eval_buffer(self):
        return True

    def eval_buffer(self, arg, res) -> "int":
        #return super().eval_buffer(*args)
        #a = np.frombuffer(arg[0], dtype=np.float64)
        #b = np.frombuffer(arg[1], dtype=np.float64)
        #c = np.frombuffer(arg[2], dtype=np.float64).reshape((3,3), order='F')
        #print(c)
        #r0 = np.frombuffer(res[0], dtype=np.float64)
        #r1 = np.frombuffer(res[1], dtype=np.float64).reshape((3,3), order='F')
        #r0[:] = np.dot(a*c,b)
        #r1[:,:] = c**2

        t0 = time.time()
        a = np.frombuffer(arg[0], dtype=np.float64).reshape((self.in_rows,-1),order='F')
        #a = np.ndarray(arg[0],dtype=np.float64,copy=False,subok=True)#.__array_custom__()
        #print(a)
        t1 = time.time()
        self.time1 += t1-t0
        return_value = self.f(a)#.reshape(self.out_rows,self.N).__array__()
        t2 = time.time()
        self.time2 += t2-t1
        r0 = np.frombuffer(res[0], dtype=np.float64).reshape((self.out_rows,self.N),order='F')
        r0[:] = np.squeeze(np.asarray(return_value))#.__array__()#.reshape(self.out_rows,self.N).__array__()
        t3 = time.time()
        self.time3 += t3-t2
        return 0

    # Evaluate numerically
    # def eval(self, arg):
        

    #     t0 = time.time()
    #     a = np.ndarray(arg[0],dtype=np.float64,copy=False,subok=True)#.__array_custom__()
    #     t1 = time.time()
    #     self.time1 += t1-t0
    #     return_value = self.f(a)#.reshape(self.out_rows,self.N).__array__()
    #     t2 = time.time()
    #     self.time2 += t2-t1
    #     return_value = np.squeeze(np.asarray(return_value))#.__array__()#.reshape(self.out_rows,self.N).__array__()
    #     t3 = time.time()
    #     self.time3 += t3-t2



        # self.its+=1
        # return [return_value]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        #print(self.name_in)
        #print(self.name_out)

        new_opts={}
        new_opts['in_dim'] = [[self.in_rows,self.N],[self.out_rows,self.N]]
        new_opts['out_dim'] = [[self.in_rows,self.out_rows*self.N]] 
        callback = ReverseFunJacbian(name,f = self.f, opts=new_opts)
        return callback
        #self.refs.append(callback)
        #nominal_in = self.mx_in()        
        #nominal_out = self.mx_out()
        #adj_seed = self.mx_out()
        #casadi_bal = callback.call(nominal_in)
        #casadi_bal = [bal.T() for bal in casadi_bal]
        #out = ca.horzcat(*casadi_bal)
        #return ca.Function(name, nominal_in + nominal_out, casadi_bal, inames, onames)
        #return callable



def siso_test():
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

def mimo_test():
    def f_test(x):
        #x = jnp.reshape(x,(-1,))
        return jnp.asarray([x[1]*x[4],x[0]*x[3], 5*x[2]+x[4], 4*x[1]**2+x[4]*x[5] - 2*x[2], jnp.exp(x[6]) + x[2] * jnp.sin(x[0]),x[1]*x[5]]).reshape((-1,))
    #f_test = lambda x: 
    f = jax.jit(jax.vmap(f_test,in_axes=-1,out_axes=-1))


    dm_x = ca.DM_rand(7,2)
    jnp_x = jnp.asarray(dm_x,dtype=jnp.float32)   
    print(f(jnp_x))
    #opts={'in_dim':[[3,1],[4,1]],'out_dim':[[5,1]],'n_in':2,'n_out':1}
    evaluator1 = JaxCasadiCallbackMIMO('f1',f_test,in_rows=7,out_rows=6,N=2)
     

    print('---------------------------------')
    print('--------------------------------')

    z_callback = evaluator1(dm_x)
    print('callback vjp evaluate:')
    print(z_callback)

    x = ca.MX.sym("x",7,1)
    z = ca.vertcat(x[0]*x[3], 5*x[2]+x[4], 4*x[1]**2+x[4]*x[5] - 2*x[2], ca.exp(x[6]) + x[2] * ca.sin(x[0]),x[1]*x[5])
    F = ca.Function('F', [x], [z])
    z_casadi = F(dm_x)
    print('casadi original evaluate:')
    print(z_casadi)


    J_callback = ca.Function('J1', [x], [ca.jacobian(evaluator1(x), x)])
    jac_callback = J_callback(dm_x)
    print('callback vjp jacobian:')
    print(jac_callback)
    print(evaluator1.ref[0].time)


    J_casadi = ca.Function('j_casadi', [x], [ca.jacobian(F(x), x)])
    jac_casadi = J_casadi(dm_x)
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

if __name__ =='__main__':
    mimo_test()