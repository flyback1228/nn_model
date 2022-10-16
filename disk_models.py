import re
import casadi as ca
import numpy as np
import timeit
import tensorflow as tf


theta = theta_1 = theta_2 = theta_3 = 2.25e-4
c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
d = np.array([6.78,  8.01,  8.82])*1e-5
tau = 1e-2

def model(x, u):
    phi_1 = x[0, :]
    phi_2 = x[1, :]
    phi_3 = x[2, :]
    dphi_1 = x[3, :]
    dphi_2 = x[4, :]
    dphi_3 = x[5, :]
    phi_1_m = x[6, :]
    phi_2_m = x[7, :]

    phi_m_1_set = u[0, :]
    phi_m_2_set = u[1, :]
    
    return ca.vertcat(
        dphi_1,
        dphi_2,
        dphi_3,
        -c[0]/theta_1*(phi_1-phi_1_m)-c[1]/theta_1 *
        (phi_1-phi_2)-d[0]/theta_1*dphi_1,
        -c[1]/theta_2*(phi_2-phi_1)-c[2]/theta_2 *
        (phi_2-phi_3)-d[1]/theta_2*dphi_2,
        -c[2]/theta_3*(phi_3-phi_2)-c[3]/theta_3 *
        (phi_3-phi_2_m)-d[2]/theta_3*dphi_3,
        1/tau*(phi_m_1_set - phi_1_m),
        1/tau*(phi_m_2_set - phi_2_m)
    )


def tf_model(x, u):
    phi_1 = x[0, :]
    phi_2 = x[1, :]
    phi_3 = x[2, :]
    dphi_1 = x[3, :]
    dphi_2 = x[4, :]
    dphi_3 = x[5, :]
    phi_1_m = x[6, :]
    phi_2_m = x[7, :]

    phi_m_1_set = u[0, :]
    phi_m_2_set = u[1, :]
    
    return tf.concat([
        dphi_1,
        dphi_2,
        dphi_3,
        -c[0]/theta_1*(phi_1-phi_1_m)-c[1]/theta_1 *
        (phi_1-phi_2)-d[0]/theta_1*dphi_1,
        -c[1]/theta_2*(phi_2-phi_1)-c[2]/theta_2 *
        (phi_2-phi_3)-d[1]/theta_2*dphi_2,
        -c[2]/theta_3*(phi_3-phi_2)-c[3]/theta_3 *
        (phi_3-phi_2_m)-d[2]/theta_3*dphi_3,
        1/tau*(phi_m_1_set - phi_1_m),
        1/tau*(phi_m_2_set - phi_2_m)],0
    )

class DynamicsModel(ca.Callback):
    def __init__(self, name, opts={}):
        ca.Callback.__init__(self)
        self.construct(name, opts)        
    # Number of inputs and outputs
    def get_n_in(self): return 2
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return ca.Sparsity.dense(8, 1)
        elif i == 1:
            return ca.Sparsity.dense(2, 1)
        else:
            return ca.Sparsity.dense(8, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(8, 1)

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        phi_1, phi_2, phi_3, dphi_1, dphi_2, dphi_3, phi_1_m, phi_2_m = ca.vertsplit(
            arg[0])
        phi_m_1_set, phi_m_2_set = ca.vertsplit(arg[1])

        return [ca.vertcat(
            dphi_1,
            dphi_2,
            dphi_3,
            -c[0]/theta*(phi_1-phi_1_m)-c[1] /theta*(phi_1-phi_2)-d[0]/theta*dphi_1,
            -c[1]/theta*(phi_2-phi_1)-c[2] / theta*(phi_2-phi_3)-d[1]/theta*dphi_2,
            -c[2]/theta*(phi_3-phi_2)-c[3]/theta *(phi_3-phi_2_m)-d[2]/theta*dphi_3,
            1/tau*(phi_m_1_set - phi_1_m),
            1/tau*(phi_m_2_set - phi_2_m)
        )]



class DynamicsModelWithForward(DynamicsModel):
    def has_forward(self, nfwd):
        return nfwd==1
    
    def get_forward(self,nfwd,name,inames,onames,opts):
        print(nfwd)
        class ForwardFun(ca.Callback):
            def __init__(self, name,opts={}):
                ca.Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 5
            def get_n_out(self): return 1

            def get_sparsity_in(self,i):
                if(i==0 or i==2):
                    return ca.Sparsity.dense(8,1)
                elif(i==1):
                    return ca.Sparsity.dense(2,1)
                elif(i==3):
                    return ca.Sparsity.dense(8,1)
                elif(i==4):
                    return ca.Sparsity.dense(2,1)

            def get_sparsity_out(self,i):
                # Forward sensitivity
                return ca.Sparsity.dense(8,1)

            # Evaluate numerically
            def eval(self, arg):
                phi_1_dot,phi_2_dot,phi_3_dot,dphi_1_dot, dphi_2_dot, dphi_3_dot, phi_1_m_dot, phi_2_m_dot = ca.vertsplit(arg[3])
                #print("Forward sweep with", arg[3])
                r0_dot = 1*dphi_1_dot
                r1_dot = 1*dphi_2_dot
                r2_dot = 1*dphi_3_dot
                r3_dot = -c[0]/theta*(phi_1_dot-phi_1_m_dot)-c[1]/theta*(phi_1_dot-phi_2_dot)-d[0]/theta*dphi_1_dot
                r4_dot = -c[1]/theta*(phi_2_dot-phi_1_dot)-c[2]/theta*(phi_2_dot-phi_3_dot)-d[1]/theta*dphi_2_dot
                r5_dot = -c[2]/theta*(phi_3_dot-phi_2_dot)-c[3]/theta *(phi_3_dot-phi_2_m_dot)-d[2]/theta*dphi_3_dot
                r6_dot = 1/tau*(0 - phi_1_m_dot)
                r7_dot = 1/tau*(0 - phi_2_m_dot)

        
                ret = ca.vertcat(r0_dot,r1_dot,r2_dot,r3_dot,r4_dot,r5_dot,r6_dot,r7_dot)
                return [ret]
                
        # You are required to keep a reference alive to the returned Callback object
        self.fwd_callback = ForwardFun(name)
        return self.fwd_callback


class DynamicsModelWithReverse(DynamicsModel):
    def has_reverse(self, nfwd):
        return nfwd==1
    
    def get_reverse(self,nfwd,name,inames,onames,opts):
        print(nfwd)
        class ReverseFun(ca.Callback):
            def __init__(self, name,opts={}):
                ca.Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 4
            def get_n_out(self): return 2

            def get_sparsity_in(self,i): 
                if(i==1):
                    return ca.Sparsity.dense(2,1)
                else:
                    return ca.Sparsity.dense(8,1)

            def get_sparsity_out(self,i):
                # Forward sensitivity
                if(i==0):
                    return ca.Sparsity.dense(8,1)
                else:
                    return ca.Sparsity.dense(2,1)

            # Evaluate numerically
            def eval(self, arg):
                #print(arg)
                #phi_1,phi_2,phi_3,dphi_1, dphi_2, dphi_3, phi_1_m, phi_2_m = ca.vertsplit(arg[3])
                bar_0,bar_1,bar_2,bar_3, bar_4, bar_5, bar_6, bar_7 = ca.vertsplit(arg[3])
                

                # dphi_1,
                # dphi_2,
                # dphi_3,
                # -c[0]/theta*(phi_1-phi_1_m)-c[1] /theta*(phi_1-phi_2)-d[0]/theta*dphi_1,
                # -c[1]/theta*(phi_2-phi_1)-c[2] /theta*(phi_2-phi_3)-d[1]/theta*dphi_2,
                # -c[2]/theta*(phi_3-phi_2)-c[3]/theta *(phi_3-phi_2_m)-d[2]/theta*dphi_3,
                # 1/tau*(phi_m_1_set - phi_1_m),
                # 1/tau*(phi_m_2_set - phi_2_m)
            
                r0_dot = bar_3*(-c[0]/theta-c[1]/theta) + bar_4*c[1]/theta
                r1_dot = bar_3* c[1] /theta +bar_4*(-c[1]/theta-c[2] / theta)+bar_5*c[2]/theta
                r2_dot = bar_4*(c[2] / theta)+ bar_5*(-c[2]/theta-c[3]/theta)
                r3_dot = bar_0 + bar_3 *(-d[0]/theta)
                r4_dot = bar_1 + bar_4 *(-d[1]/theta)
                r5_dot = bar_2 + bar_5 *(-d[2]/theta)
                r6_dot = bar_3*c[0]/theta-bar_6/tau
                r7_dot = bar_5*c[3]/theta-bar_7/tau
                ret_0 = ca.vertcat(r0_dot,r1_dot,r2_dot,r3_dot,r4_dot,r5_dot,r6_dot,r7_dot)
                ret_1 = ca.vertcat(-bar_6/tau,-bar_7/tau)
                return [ret_0,ret_1]
                
        # You are required to keep a reference alive to the returned Callback object
        self.rev_callback = ReverseFun(name)
        return self.rev_callback


if __name__ == '__main__':
    x0 = np.pi*np.array([1, 1, -1.5, 1, -1, 1, 0, 0]).reshape(-1, 1)
    u0 = np.pi*np.array([-1, 0.2]).reshape(-1, 1)
    dynamics_model = DynamicsModel('model')    

    print(dynamics_model(x0, u0))
    print(model(x0, u0))

    dynamics_model_with_forward = DynamicsModelWithForward('model_with_forward')
    dynamics_model_with_reverse = DynamicsModelWithReverse('model_with_reverse')
    x = ca.MX.sym("x",8,1)
    u = ca.MX.sym("x",2,1)
        
    f = ca.Function('f',[x,u],[model(x,u)])
    J0 = ca.Function('J0',[x,u],[ca.jacobian(f(x,u),u)])
    print(J0(x0,u0))

    J1 = ca.Function('J1',[x,u],[ca.jacobian(dynamics_model_with_forward(x,u),x)])
    print(J1(x0,u0))

    J2 = ca.Function('J2',[x,u],[ca.jacobian(dynamics_model_with_reverse(x,u),x)])
    print(J2(x0,u0))


    #x0 = ca.pi*ca.DM([[1, 1, -1.5, 1, -1, 1, 0, 0],[1, 1, -1.5, 1, -1, 1, 0, 0]]).T
    #u0 = ca.pi*ca.DM([[-1, 0.2],[-5, 0.3]]).T
    #print(x0)
    #print(u0)
    #print(dynamics_model_with_forward(x0, u0))
    #print(model(x0, u0))
    #print(J0(x0,u0))
    #print(J1(x0,u0))
