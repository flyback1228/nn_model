from gc import callbacks
from casadi import *
#
# How to use Callback
# Joel Andersson
#


class MiMoCallback(Callback):
	def __init__(self, name, opts={}):
		Callback.__init__(self)
		self.name = name
		self.construct(name, opts)

    # Number of inputs and outputs
	def get_n_in(self): return 2
	def get_n_out(self): return 3

	def get_sparsity_in(self, i):
		if (i == 0):
			return Sparsity.dense(2, 1)
		else:
			return Sparsity.dense(1, 1)

	def get_sparsity_out(self, i):
		if (i == 0):
			return Sparsity.dense(2, 1)
		else:
			return Sparsity.dense(1, 1)

    # Initialize the object
	def init(self):
		print('initializing object')

    # Evaluate numerically
	def eval(self, arg):
		x, y = vertsplit(arg[0])
		z = arg[1]
		return [vertcat(y*exp(x), 5*x**2+y**2+z), 3*x*y**2*z, 5*sin(x)+6*cos(y)+sin(z)]

	def has_reverse(self, nadj):
		return nadj==1

	def get_reverse(self, nfwd, name, inames, onames, opts):
		# You are required to keep a reference alive to the returned Callback object
		class ReverseCallback(Callback):
			def __init__(self, name, opts={}):
				Callback.__init__(self)
				self.construct(name, opts)

			def get_n_in(self): return 8
			def get_n_out(self): return 2

			def get_sparsity_in(self, i):
				if (i == 1 or i == 3 or i == 4 or i==7):  # nominal input
					print(i)
					return Sparsity.dense(1, 1)
				elif (i == 0 or i == 2):  # nominal output
					print(i)
					return Sparsity.dense(2, 1)
				elif (i == 5):  # nominal output
					print(i)
					return Sparsity.dense(2, 1)
				elif (i == 6):
					print(i)
					return Sparsity.dense(1, 1)


			def get_sparsity_out(self, i):
				# Reverse sensitivity\
				if i==0:
					return Sparsity.dense(2, 1)
				else:
					return Sparsity.dense(1,1)

			# Evaluate numerically
			def eval(self, arg):				
				print(arg)
				#return [vertcat(y*exp(x), 5*x**2+y**2+z), 3*x*y**2*z, 5*sin(x)+6*cos(y)+sin(z)]
				bar_0 ,bar_1 = vertsplit(arg[5])
				bar_2 = arg[6]
				bar_3 = arg[7]
				x,y = vertsplit(arg[0])
				z = arg[1]
				x_bar = dot(DM([y*exp(x),10*x,3*y**2*z,6*cos(x)]),DM([bar_0,bar_1,bar_2,bar_3]))
				y_bar = dot(DM([exp(x),2*x,6*x*y*z,-6*sin(y)]),DM([bar_0,bar_1,bar_2,bar_3]))
				z_bar = dot(DM([0,1,3*x*y**2,6*cos(z)]),DM([bar_0,bar_1,bar_2,bar_3]))

				return [DM([x_bar,y_bar]),z_bar]
		self.rev_callback = ReverseCallback(name)
		return self.rev_callback

f = MiMoCallback('f')
res = f(DM([2,1]),3)
print(res)

x = MX.sym("x",2,1)
z = MX.sym("z")
J = Function('J', [x,z], [jacobian(f(x,z)[1], x)])
print(J(DM([2,1]),3))


class MyCallback(Callback):
    def __init__(self, name, d, opts={}):
        Callback.__init__(self)
        self.d = d
        self.construct(name, opts)

    # Number of inputs and outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1
	

    # Initialize the object
    def init(self):
        print('initializing object')

    # Evaluate numerically
    def eval(self, arg):
        x = arg[0]
        f = sin(self.d*x)
        return [f]


# Use the function
f = MyCallback('f', 0.5)
res = f(2)
print(res)
def has_reverse(self, nadj):
		return nadj==1
# You may call the Callback symbolically
x = MX.sym("x")
print(f(x))

# Derivates OPTION 1: finite-differences
eps = 1e-5
print((f(2+eps)-f(2))/eps)

f = MyCallback('f', 0.5, {"enable_fd": True})
J = Function('J', [x], [jacobian(f(x), x)])
print(J(2))

# Derivates OPTION 2: Supply forward mode
# Example from https://www.youtube.com/watch?v=mYOkLkS5yqc&t=4s


class Example4To3(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return Sparsity.dense(4, 1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(3, 1)

    # Evaluate numerically
    def eval(self, arg):
        a, b, c, d = vertsplit(arg[0])
        ret = vertcat(sin(c)*d+d**2, 2*a+c, b**2+5*c)
        return [ret]


class Example4To3_Fwd(Example4To3):
    def has_forward(self, nfwd):
        # This example is written to work with a single forward seed vector
        # For efficiency, you may allow more seeds at once
        return nfwd == 1

    def get_forward(self, nfwd, name, inames, onames, opts):

        class ForwardFun(Callback):
            def __init__(self, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 3
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if i == 0:  # nominal input
                    return Sparsity.dense(4, 1)
                elif i == 1:  # nominal output
                    return Sparsity(3, 1)
                else:  # Forward seed
                    return Sparsity.dense(4, 1)

            def get_sparsity_out(self, i):
                # Forward sensitivity
                return Sparsity.dense(3, 1)

            # Evaluate numerically
            def eval(self, arg):
                # vertcat(sin(c)*d+d**2,2*a+c,b**2+5*c)
                print(len(arg))
                a, b, c, d = vertsplit(arg[0])
                print("Forward arg_0 ", a, b, c, d)
                a_dot, b_dot, c_dot, d_dot = vertsplit(arg[2])
                print("Forward sweep with", a_dot, b_dot, c_dot, d_dot)
                # w0 = sin(c)
                # w0_dot = cos(c)*c_dot
                # w1 = w0*d
                # w1_dot = w0_dot*d+w0*d_dot
                # w2 = d**2
                # w2_dot = 2*d_dot*d
                # r0 = w1+w2
                r0_dot = cos(c)*c_dot*d+sin(c)*d_dot + 2*d_dot*d
                # w3 = 2*a
                # w3_dot = 2*a_dot
                # r1 = w3+c
                r1_dot = 2*a_dot+c_dot
                # w4 = b**2
                # w4_dot = 2*b_dot*b
                # w5 = 5*sin(c)
                # w5_dot = 5*cos(c)*c_dot
                # r2 = w4+w5
                r2_dot = 2*b_dot*b + 5*cos(c)*c_dot
                ret = vertcat(r0_dot, r1_dot, r2_dot)
                return [ret]
        # You are required to keep a reference alive to the returned Callback object
        self.fwd_callback = ForwardFun()
        return self.fwd_callback


f = Example4To3_Fwd('f')
x = MX.sym("x", 4, 2)
J = Function('J', [x], [jacobian(f(x), x)])
a = DM([[1, 2, 0, 3], [3, 4, 5, 7]])
print(vertsplit(a))
# print(f(a))
print(J(a.T))
# print(J(vertcat(1,2,0,3)))

# Derivates OPTION 3: Supply reverse mode


class Example4To3_Rev(Example4To3):
    def has_reverse(self, nadj):
        # This example is written to work with a single forward seed vector
        # For efficiency, you may allow more seeds at once
        return nadj == 1

    def get_reverse(self, nfwd, name, inames, onames, opts):

        class ReverseFun(Callback):
            def __init__(self, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 3
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if i == 0:  # nominal input
                    print(i)
                    return Sparsity.dense(4, 1)
                elif i == 1:  # nominal output
                    print(i)
                    return Sparsity.dense(3, 1)
                else:  # Reverse seed
                    print(i)
                    return Sparsity.dense(3, 1)

            def get_sparsity_out(self, i):
                # Reverse sensitivity
                return Sparsity.dense(4, 1)

            # Evaluate numerically
            def eval(self, arg):
                # vertcat(sin(c)*d+d**2,2*a+c,b**2+5*c)
                print(arg)
                a, b, c, d = vertsplit(arg[0])
                r0_bar, r1_bar, r2_bar = vertsplit(arg[2])
                print("Reverse sweep with", r0_bar, r1_bar, r2_bar)

                b_bar = 2*b*r2_bar
                a_bar = 2*r1_bar
                d_bar = (sin(c)+2*d)*r0_bar
                c_bar = r1_bar + cos(c)*d*r0_bar+5*r1_bar
                ret = vertcat(a_bar, b_bar, c_bar, d_bar)
                return [ret]
        # You are required to keep a reference alive to the returned Callback object
        self.rev_callback = ReverseFun()
        return self.rev_callback


f = Example4To3_Rev('f')
x = MX.sym("x", 4)
J = Function('J', [x], [jacobian(f(x), x)])
print(J(vertcat(1, 2, 0, 3)))

# Derivates OPTION 4: Supply full Jacobian


class Example4To3_Jac(Example4To3):
    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        class JacFun(Callback):
            def __init__(self, opts={}):
                Callback.__init__(self)
                self.construct(name, opts)

            def get_n_in(self): return 2
            def get_n_out(self): return 1

            def get_sparsity_in(self, i):
                if i == 0:  # nominal input
                    return Sparsity.dense(4, 1)
                elif i == 1:  # nominal output
                    return Sparsity(3, 1)

            def get_sparsity_out(self, i):
                return sparsify(DM([[0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0]])).sparsity()

            # Evaluate numerically
            def eval(self, arg):
                print(len(arg))
                a, b, c, d = vertsplit(arg[0])
                ret = DM(3, 4)
                ret[0, 2] = d*cos(c)
                ret[0, 3] = sin(c)+2*d
                ret[1, 0] = 2
                ret[1, 2] = 1
                ret[2, 1] = 2*b
                ret[2, 2] = 5
                return [ret]

        # You are required to keep a reference alive to the returned Callback object
        self.jac_callback = JacFun()
        return self.jac_callback


f = Example4To3_Jac('f')
x = MX.sym("x", 4)
J = Function('J', [x], [jacobian(f(x), x)])
print(J(vertcat(1, 2, 0, 3)))
