import jax.numpy as jnp
from jax import grad
from jax import random as jrandom
import time
from tqdm import tqdm 

from jax import jit, grad, jacfwd
from scipy.interpolate import Rbf, RBFInterpolator


from jax.config import config
config.update("jax_enable_x64", True)

class OptimizationBlueprint:
    def __init__(self, x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps=0, verbose=False, x_opt=None):
        self.jrandom_key = jrandom_key

        self.step_size = step_size

        self.sig = sig

        self.F = F

        self.loop_steps_remaining = num_total_steps
        self.num_total_steps = num_total_steps
        self.grad_eps = grad_eps

        self.verbose = True
        self.x_init = x_init
        self.dim = len(x_init)
        self.verbose = verbose

        self.x_opt = x_opt
         

    def run_opt(self):
        X = self.x_init

        vals_arr = []
        x_arr = []
        total_func_calls = 0
        start_time = time.time()

        vals_arr.append((self.F.f(X), time.time() - start_time, total_func_calls, 0))
        x_arr.append(X)

        for t in tqdm(range(self.loop_steps_remaining)):

            
            # get search direction
            if self.jrandom_key is not None:
                self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
                search_direction, f1, num_func_calls = self.step_getter(X, subkey)
            else:
                search_direction, f1, num_func_calls = self.step_getter(X)
            total_func_calls += num_func_calls

            vals_arr.append((self.F.f(X), time.time() - start_time, total_func_calls, float(jnp.linalg.norm(f1 - self.F.f1(X))))) # jnp.linalg.norm(X - self.F.x_opt))) #float(f1.T @ self.F.f1(X)) / (jnp.linalg.norm(self.F.f1(X)) * jnp.linalg.norm(f1))))#  float(jnp.linalg.norm(f1 - self.F.f1(X)))/jnp.linalg.norm(self.F.f1(X))))# jnp.linalg.norm(X - self.x_init)))# # jnp.linalg.norm(self.F.f1(X)))) # #float(jnp.linalg.norm(self.grad_curr - self.F.f1(X))/jnp.linalg.norm(self.F.f1(X))))) #/jnp.linalg.norm(self.F.f1(X))))) jnp.linalg.norm(alpha * search_direction))) #
            x_arr.append(X)

            if jnp.linalg.norm(f1)/self.dim < self.grad_eps:
                break

            if self.verbose:
                # print(X)
                print("Number Iterations", t)
                print("Num Function Calls", total_func_calls)
                print("Obj", self.F.f(X))
                print("Grad norm", jnp.linalg.norm(f1))
                print("True Norm", jnp.linalg.norm(self.F.f1(X)))
                print()

            if self.F.f(X) == float("inf"):
                break
            
            # update step
            X = X + self.step_size * search_direction

            if vals_arr[-1][-1] > 1e10:
                break

        return X, jnp.array(vals_arr), jnp.array(x_arr)


    def step_getter(self, X, jrandom_key=None):
        pass


class NewtonMethod(OptimizationBlueprint):
    def __init__(self, x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps=0, verbose=False):
        super().__init__(x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps, verbose)
        self.sig = 0
        self.grad_curr = None
    
    def step_getter(self, X, jrandom_key=None):
        f1 = self.F.f1(X)
        f2 = self.F.f2(X)
        self.grad_curr = f1
        return -jnp.linalg.inv(f2).dot(f1), f1, 2

class GradientDescent(OptimizationBlueprint):
    def __init__(self, x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps=0, verbose=False):
        super().__init__(x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps, verbose)
        self.sig = 0
        self.grad_curr = None
    
    def step_getter(self, X, jrandom_key=None):
        f1 = self.F.f1(X)
        self.grad_curr = f1
        return -f1, f1, 2 


class ExactH_GD(OptimizationBlueprint):
    def __init__(self, x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps=0, verbose=False):
        super().__init__(x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps, verbose)

        self.grad_getter = grad_getter
        

    def step_getter(self, X, jrandom_key):
        num_func_calls = 0
        
        H = self.F.f2(X)
        f1, num_func_calls, _, _, _ = self.grad_getter.grad(self.F, X, jrandom_key, H=H)

        if self.verbose:
            print("Grad diff", jnp.linalg.norm(f1 - self.F.f1(X))/jnp.linalg.norm(self.F.f1(X)))

        return -f1, f1, num_func_calls





class InterpH_GD(OptimizationBlueprint):
    def __init__(self, x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps=0, verbose=False, smoothing=1):
        super().__init__(x_init, F, step_size, num_total_steps, sig, jrandom_key, grad_eps, verbose)


        self.X_prev = None
        self.grad_curr = None
        self.grad_getter = grad_getter
        
        self.interp_points = None
        self.F_vals = None
        
        self.smoothing = smoothing
        

    def step_getter(self, X, jrandom_key):
        num_func_calls = 0
        if (self.interp_points is not None) and (self.interp_points.shape[1] > 2 * len(X) + 1):
            curr_interp_points = []
            curr_F_vals = []
            delta_up = 5
            delta_low = 1e-9
            # for i in range(self.interp_points.shape[1]):
            #     dist = jnp.linalg.norm(self.interp_points[:, i] - X)
            #     if (dist < delta_up) and (dist > delta_low):
            #         curr_interp_points.append(self.interp_points[:, i])
            #         curr_F_vals.append(self.F_vals[i])

            curr_interp_points = self.interp_points[:, -1000:] # int(-self.dim**2 * 2):]# self.interp_points # jnp.array(curr_interp_points).T # 
            curr_F_vals = self.F_vals[-1000:] # [-int(self.dim**2 * 2): ] # self.F_vals # jnp.array(curr_F_vals) # 
            # print("num F sub", len(curr_F_vals))

            # curr_interp_points = curr_interp_points[:, -int(self.dim**2/8):]
            # curr_F_vals = curr_F_vals[-int(self.dim**2/8):]

            # print("num F sub", len(curr_F_vals))
            # print("F sub all", len(self.F_vals))
            # print()

            # if curr_interp_points.shape[1] < 2 * len(X) + 1:
            #     H = jnp.eye(len(X))
            # else:
            #     H = self.get_G(curr_interp_points, curr_F_vals)

            # H = self.get_G(curr_interp_points, curr_F_vals)
            H, rbf_f1 = self.get_H_rbf(X, curr_interp_points, curr_F_vals)
            # print(repr(H))
            # print("H diff", jnp.linalg.norm(H - self.F.f2(X))/jnp.linalg.norm(self.F.f2(X)))

        else:
            H = jnp.eye(len(X))
            rbf_f1 = None

        self.grad_curr, num_func_calls, F_x_0, FS, S = self.grad_getter.grad(self.F, X, jrandom_key, H=H)

        if self.interp_points is None:
            self.interp_points = jnp.concatenate([S + X.reshape(-1, 1), X.reshape(-1, 1)], axis=1)
            self.F_vals = jnp.concatenate([FS, jnp.array([F_x_0])])
        else:
            self.interp_points = jnp.concatenate([self.interp_points, S + X.reshape(-1, 1), X.reshape(-1, 1)], axis=1)
            self.F_vals = jnp.concatenate([self.F_vals, FS, jnp.array([F_x_0])])
        
        if self.verbose:

            print("Grad diff", jnp.linalg.norm(self.grad_curr - self.F.f1(X))/jnp.linalg.norm(self.F.f1(X)))
            print("H diff", jnp.linalg.norm(H - self.F.f2(X))/jnp.linalg.norm(self.F.f2(X)))
            # print("H eigs", (jnp.linalg.eigh(H)[0] - jnp.linalg.eigh(self.F.f2(X))[0])/jnp.linalg.eigh(self.F.f2(X))[0])
            
        f1 = self.grad_curr

        return -f1, f1, num_func_calls
    
    def create_W(self, S):
        dim = S.shape[0]
        N = S.shape[1]
        A = 1/2 * (S.T @ S)**2
        eXT = jnp.concatenate([jnp.ones(shape=(N, 1)), S.T], axis=1)
        eTX0 = jnp.concatenate([eXT.T, jnp.zeros(shape=(dim+1, dim+1))], axis=1)
        W = jnp.concatenate([A, eXT], axis=1)
        W = jnp.concatenate([W, eTX0], axis=0)
        return W
    
    def get_G(self, S, F_vals):
        dim = len(S)
        
        W = self.create_W(S)
        F_vals0 = jnp.concatenate([F_vals, jnp.zeros(dim + 1)])
        lcg = jnp.linalg.solve(W, F_vals0)

        lmbda = lcg[:-(dim+1)]

        return S @ jnp.diag(lmbda) @ S.T
    
    def get_H_rbf(self, x_0, S, F_vals):

        rbf = RBFInterpolator(S.T, F_vals, smoothing=self.smoothing) #, epsilon=0.1, kernel="gaussian")
        coeffs = jnp.array(rbf._coeffs)
        y = jnp.array(rbf.y)
        epsilon = rbf.epsilon
        shift = jnp.array(rbf._shift)
        scale = jnp.array(rbf._scale)
        powers = jnp.array(rbf.powers)

        H = self._thin_plate_f2(x_0, y, epsilon, coeffs, shift, scale, powers)

        # rbf_f1 = grad(lambda x: self._evaluate(x, y, epsilon, coeffs, shift, scale, powers))
        # rbf_f2 = jacfwd(lambda x: rbf_f1(x))
        # print("H diff approx", jnp.linalg.norm(H - rbf_f2(x_0)))

        f1 = None # rbf_f1(jnp.array(x_0))


        return H, f1

    def _thin_plate_f2(self, x, y, epsilon, coeffs, shift, scale, powers):
        dim = len(x)
        p = y.shape[0]
        yeps = y*epsilon
        xeps = x*epsilon
        r = jnp.linalg.norm(xeps - yeps, axis=1)

        log_r = jnp.log(r)

        a = 2 * epsilon**2 * jnp.eye(dim) * (log_r @ coeffs[:p, 0])
        b = 2 * epsilon**2 * (xeps - yeps).T @ ((coeffs[:p, 0]/r**2).reshape(-1, 1) * (xeps - yeps))
        c = 2 * epsilon * jnp.eye(dim) * jnp.sum(coeffs[:p, 0])

        return a + b + c

