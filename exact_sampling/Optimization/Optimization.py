import jax.numpy as jnp
from jax import grad, random as jrandom
import time
from tqdm import tqdm 

from jax.config import config
config.update("jax_enable_x64", True)

class OptimizationBlueprint:
    def __init__(self, x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps=0, verbose=False, x_opt=None):
        self.jrandom_key = jrandom_key

        self.c1 = c1
        self.c2 = c2

        self.sig = sig

        self.F = F

        self.linesearch = helper_linesearch(self.F, self.c1, self.c2)
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

        for t in range(self.loop_steps_remaining): # self.loop_steps_remaining > 0:
            # self.loop_steps_remaining -= 1
            
            # get search direction
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            search_direction, f1, num_func_calls = self.step_getter(X, subkey)
            total_func_calls += num_func_calls

            if jnp.linalg.norm(f1)/self.dim < self.grad_eps:
                break

            if self.verbose:
                # print(X)
                print("Number Iterations", t)
                print("Num Function Calls", total_func_calls)
                print("Obj", self.F.f(X))
                print("Grad norm", jnp.linalg.norm(f1))
                print()
            
            # do line search
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            alpha, num_func_calls = self.linesearch(X, f1, search_direction, self.sig, subkey) 
            total_func_calls += num_func_calls

            # update step
            X = X + alpha * search_direction

            # clean up after update (i.e. BFGS update)
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            num_func_calls = self.post_step(X, subkey)
            total_func_calls += num_func_calls
            vals_arr.append((self.F.f(X), time.time() - start_time, total_func_calls, float(jnp.linalg.norm(self.grad_curr - self.F.f1(X))))) #/jnp.linalg.norm(self.F.f1(X))))) jnp.linalg.norm(alpha * search_direction))) #
            x_arr.append(X)

            if vals_arr[-1][-1] > 1e10:
                break

        self.reset()
        return X, jnp.array(vals_arr), jnp.array(x_arr)


    def step_getter(self, X, jrandom_key):
        pass

    def post_step(self, X, jrandom_key):
        return 0

    def reset(self):
        pass


class BFGS(OptimizationBlueprint):
    def __init__(self, x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps=0, verbose=False):
        super().__init__(x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose)
        # self.H_inv = jnp.linalg.inv(F.f2(x_init)) 
        # self.H_inv = jnp.diag(jnp.diag(jnp.linalg.inv(F.f2(x_init)) ))
        # print(self.H_inv)


        self.H_inv = jnp.eye(self.dim)
        # self.H_inv = jnp.linalg.inv(F.f2(x_init)) 

        self.X_prev = None
        self.grad_curr = None
        self.grad_getter = grad_getter

        self.use_exact = False


    def step_getter(self, X, jrandom_key):
        num_func_calls = 0
        self.X_prev = X.copy()
        # print("pre true", jnp.linalg.eig(jnp.linalg.inv(self.F.f2(X)))[0])
        if self.grad_curr is None:
            if self.use_exact:
                self.grad_curr, num_func_calls = self.grad_getter.grad(self.F, X, jrandom_key, H=self.F.f2(X)) 
            else:
                self.grad_curr, num_func_calls = self.grad_getter.grad(self.F, X, jrandom_key, H=jnp.linalg.inv(self.H_inv))

            # print("Grad diff", jnp.linalg.norm(self.grad_curr - self.F.f1(X)))
            # print(self.grad_curr)
            # print(self.F.f1(X))
        f1 = self.grad_curr

        return -f1, f1, num_func_calls
        # return -self.H_inv.dot(f1), f1, num_func_calls
    
    def post_step(self, X, jrandom_key):
        prev_grad = self.grad_curr

        if self.use_exact:
            self.grad_curr, num_func_calls = self.grad_getter.grad(self.F, X, jrandom_key,  H=self.F.f2(X))
        else:
            self.grad_curr, num_func_calls = self.grad_getter.grad(self.F, X, jrandom_key,  H=jnp.linalg.inv(self.H_inv))

        # print(self.grad_curr)
        # print(self.F.f1(X))
        # print("pre", jnp.linalg.eig(self.H_inv)[0]) 
        if self.use_exact:
            self.H_inv = jnp.linalg.inv(self.F.f2(X))
        else:
            self.H_inv = self.BFGS_update(self.X_prev, X, prev_grad, self.grad_curr, self.H_inv)
        # print("post", jnp.linalg.eig(self.H_inv)[0])
        # print("post true", jnp.linalg.eig(jnp.linalg.inv(self.F.f2(X)))[0])
        if self.verbose:
            print("Grad diff", jnp.linalg.norm(self.grad_curr - self.F.f1(X))/jnp.linalg.norm(self.F.f1(X)))
            print("H diff", jnp.linalg.norm(self.H_inv - jnp.linalg.inv(self.F.f2(X)))/jnp.linalg.norm(jnp.linalg.inv(self.F.f2(X))))
        return num_func_calls

    def reset(self):
        self.grad_curr = None
        self.H_inv = jnp.eye(self.dim)

    def BFGS_update(self, x_0, x_1, g_0, g_1, inv_hessian_approx=None):
        if inv_hessian_approx is None:
            H = jnp.eye(len(x_0))
        else:
            H = inv_hessian_approx

        grad_diff = (g_1 - g_0)
        update_step = x_1 - x_0
        
        ys = jnp.inner(grad_diff, update_step)
        Hy = jnp.dot(H, grad_diff)
        yHy = jnp.inner(grad_diff, Hy)
        
        H += (ys + yHy) * jnp.outer(update_step, update_step) / ys ** 2
        H -= (jnp.outer(Hy, update_step) + jnp.outer(update_step, Hy)) / ys

        H = (H + H.T) / 2.
        return H      


class NewtonMethod(OptimizationBlueprint):
    def __init__(self, x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps=0, verbose=False):
        super().__init__(x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose)
        self.sig = 0
        self.grad_curr = None
    
    def step_getter(self, X, jrandom_key=None):
        f1 = self.F.f1(X)
        f2 = self.F.f2(X)
        self.grad_curr = f1
        return -jnp.linalg.inv(f2).dot(f1), f1, 2

class GradientDescent(OptimizationBlueprint):
    def __init__(self, x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps=0, verbose=False):
        super().__init__(x_init, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose)
        self.sig = 0
        self.grad_curr = None
    
    def step_getter(self, X, jrandom_key=None):
        f1 = self.F.f1(X)
        self.grad_curr = f1
        return -f1, f1, 2 


def helper_linesearch(F, c1, c2):

    def helper(x_0, f1_x_0, search_direction, sig, jrandom_key):
        num_func_calls = 0
        num_func_call_limit = 100

        f0 = F.f(x_0) # we do not count this call, because it has already been done in the grad_stepper.
        dg = jnp.inner(search_direction, f1_x_0)

        if dg >= - sig * jnp.linalg.norm(search_direction):
            c1_is_zero = 0
        else:
            c1_is_zero = 1

        def armijo_rule(alpha, c1_is_zero, is_sig_term, jrandom_key):
            return F.f(x_0 + alpha * search_direction, jrandom_key) > f0 + c1_is_zero * c1*alpha*dg + is_sig_term * 2 * sig
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        jrandom_key, subkey = jrandom.split(jrandom_key)
        is_sig_term = 0
        while armijo_rule(alpha, c1_is_zero, is_sig_term, subkey) and num_func_calls < num_func_call_limit:
            num_func_calls += 1
            jrandom_key, subkey = jrandom.split(jrandom_key)
            alpha = armijo_update(alpha)
            is_sig_term = 1

        num_func_calls += 1

        return alpha, num_func_calls

    return helper



