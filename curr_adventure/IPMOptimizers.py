import jax.numpy as jnp
from jax import random as jrandom
import jax
from functools import partial
from jax import lax
import time, sys
import pickle
from .Functions import LinearCombination, get_obj
from .Barriers import get_barrier
from .derivative_estimation.utils import OneEZero
import os, psutil
process = psutil.Process(os.getpid())

def get_optimizer(config):
    if "Newton_IPM" == config["optimization_name"]:
        return Newton_IPM(config)
    elif "BFGS" == config["optimization_name"]:
        return BFGS(config)
    elif "Gradient_Descent" == config["optimization_name"]:
        return Gradient_Descent(config)
    else:
        raise Exception("{} not implemented.".dot(config["optimization_name"]))

class OptimizationBlueprint:
    def __init__(self, config):
        self.jrandom_key = jrandom.PRNGKey(config["optimization_meta"]["jrandom_key"])

        self.config = config
        self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
        self.barrier = get_barrier(config, subkey)
        self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
        self.obj = get_obj(config, subkey)

        self.dim = config["dim"]

        self.c1 = config["optimization_meta"]["c1"]
        self.c2 = config["optimization_meta"]["c2"]

        self.delta = config["optimization_meta"]["delta"]

        self.linesearch = helper_linesearch(self.obj, self.barrier, self.c1, self.c2)
        self.loop_steps_remaining = config["num_total_steps"]

        self.verbose = True


    def update(self, X, time_step, record_vals=False, record_full_path=False):
        assert not (record_vals and record_full_path)

        t = 4 * (0.5)**(time_step) 

        self.combined_F = LinearCombination(self.obj, self.barrier, [1, t])

        if record_full_path:
            full_path_arr = [(X.copy(), time.time())]
        if record_vals:
            vals_arr = [(self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time())]
        
        while self.loop_steps_remaining > 0:
            self.loop_steps_remaining -= 1
             
            # get search direction
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            f1 = self.combined_F.f1(X) # TODO: This is exact right now, should it be noisy? come from FD approx? 
            search_direction = self.step_getter(X, subkey, t)
            newton_decrement_squared = -f1[0].dot(search_direction)
            
            # check if valid search direction
            if newton_decrement_squared < 0:
                if record_full_path:
                    full_path_arr.append((X.copy(), self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time()))
                if record_vals:
                    vals_arr.append((self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time()))
                self.reset()
                continue
            newton_decrement = jnp.sqrt(newton_decrement_squared)

            if self.verbose:
                print("Newton Decrement Squared", newton_decrement_squared)
                print("Obj", self.obj.f(X))
                print("Steps Remaining", self.loop_steps_remaining)
                print()

            # Check if completed
            if newton_decrement**2 < self.delta:
                break

            # do line search
            alpha = self.linesearch(X[0], 1/(1 + newton_decrement) * search_direction, t) 

            # update step
            X[0] = X[0] + 1/(1 + newton_decrement) * alpha * search_direction
            if record_full_path:
                full_path_arr.append((X.copy(), time.time()))
            if record_vals:
                vals_arr.append((self.obj.f(X)[0], self.barrier.f(X)[0], self.combined_F.f(X)[0], time.time()))
            # clean up after update (i.e. BFGS update)
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            self.post_step(X, subkey, t)

        self.reset()
        if record_full_path:
            return X, full_path_arr
        if record_vals:
            return X, vals_arr
        return X, None


    def step_getter(self, X, jrandom_key, t):
        pass

    def post_step(self, X, jrandom_key, t):
        pass

    def reset(self):
        pass

class Newton_IPM(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.d_prime = config["optimization_meta"]["d_prime"]

    def step_getter(self, X, jrandom_key, t):
        jrandom_key, subkey = jrandom.split(jrandom_key)
    
        H_inv = self.combined_F.f2_inv(X)[0]
        f1 = self.combined_F.f1(X, subkey)[0]

        search_direction = -H_inv.dot(f1)

        return search_direction

class BFGS(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
        self.H_inv = jnp.eye(config["dim"])
        self.X_prev = None
        self.grad_curr = None
        self.grad_getter = OneEZero(self.config)

    def step_getter(self, X, jrandom_key, t):
        self.X_prev = X[0].copy()
        if self.grad_curr is None or self.config["grad_estimate_type"] == "FD_Hess":
            self.grad_curr = self.grad_getter.f1(self.combined_F, X, self.config, jrandom_key, M=self.H_inv)
        f1 = self.grad_curr
        return -self.H_inv.dot(f1)
    
    def post_step(self, X, jrandom_key, t):
        prev_grad = self.grad_curr
        self.grad_curr = self.grad_getter.f1(self.combined_F, X, self.config, jrandom_key, M=self.H_inv)
        self.H_inv = self.BFGS_update(self.combined_F, self.X_prev, X[0], prev_grad, self.grad_curr, self.H_inv)

    def reset(self):
        self.grad_curr = None
        self.H_inv = jnp.eye(self.config['dim'])

    def BFGS_update(self, F, x_0, x_1, g_0, g_1, inv_hessian_approx=None):
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
        return H    


class Gradient_Descent(OptimizationBlueprint):
    def __init__(self, config):
        super().__init__(config)
    
    def step_getter(self, X, jrandom_key, t):
        return -self.combined_F.f1(X)[0]


def helper_linesearch(obj, barrier, c1, c2):

    def helper(x_0, search_direction, t):
        combined_F = LinearCombination(obj, barrier, [1, t])
        f0 = combined_F.f(x_0.reshape(1, -1))[0]
        f1 = combined_F.f1(x_0.reshape(1, -1))[0]
        dg = jnp.inner(search_direction, f1)

        def armijo_rule(alpha):
            return combined_F.f((x_0 + alpha * search_direction).reshape(1, -1))[0] > f0 + c1*alpha*dg
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        while armijo_rule(alpha):
            alpha = armijo_update(alpha)

        return alpha

    return helper
