from tkinter import Y
import jax.numpy as jnp
from jax import random as jrandom
import time
from archive.optimize_USG_error_old import optimize_uncentered_S
from Functions import Brown

import matplotlib.pyplot as plt
import pickle 

from jax.config import config
config.update("jax_enable_x64", True)

class OptimizationBlueprint:
    def __init__(self, x_init, F, c1, c2, num_total_steps, jrandom_key):
        self.jrandom_key = jrandom_key

        self.c1 = c1
        self.c2 = c2

        self.F = F

        self.linesearch = helper_linesearch(self.F, self.c1, self.c2)
        self.loop_steps_remaining = num_total_steps

        self.verbose = True
        self.x_init = x_init
        self.dim = len(x_init)


    def run_opt(self, record_vals=False, record_full_path=False):
        assert not (record_vals and record_full_path)
        X = self.x_init

        if record_full_path:
            full_path_arr = [(X.copy(), self.F.f(X), time.time())]
        if record_vals:
            # vals_arr = [(self.F.f(X), time.time())]
            vals_arr = [(jnp.linalg.norm(self.opt_x - X), time.time())]

        
        while self.loop_steps_remaining > 0:
            self.loop_steps_remaining -= 1
             
            # get search direction
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            search_direction, f1 = self.step_getter(X, subkey)
            newton_decrement_squared = -f1.dot(search_direction)
            
            # check if valid search direction
            if newton_decrement_squared < 0:
                if record_full_path:
                    full_path_arr.append((X.copy(), self.F.f(X), time.time()))
                if record_vals:
                    # vals_arr.append((self.F.f(X), time.time()))
                    vals_arr.append((jnp.linalg.norm(self.opt_x - X), time.time()))
                continue

            if self.verbose:
                print("Newton Decrement Squared", newton_decrement_squared)
                print("Obj", self.F.f(X))
                print("Steps Remaining", self.loop_steps_remaining)
                print("Dist", jnp.linalg.norm(self.opt_x - X))
                print()

            # do line search
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            alpha = self.linesearch(X, f1, search_direction, subkey) 

            # update step
            X = X + alpha * search_direction
            if record_full_path:
                full_path_arr.append((X.copy(), time.time()))
            if record_vals:
                # vals_arr.append((self.F.f(X), time.time()))
                vals_arr.append((jnp.linalg.norm(self.opt_x - X), time.time()))


            # clean up after update (i.e. BFGS update)
            self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
            self.post_step(X, subkey)

        self.reset()
        if record_full_path:
            return X, full_path_arr
        if record_vals:
            return X, vals_arr
        return X, None


    def step_getter(self, X, jrandom_key):
        pass

    def post_step(self, X, jrandom_key):
        pass

    def reset(self):
        pass


class BFGS(OptimizationBlueprint):
    def __init__(self, x_init, F, c1, c2, num_total_steps, jrandom_key, grad_getter):
        super().__init__(x_init, F, c1, c2, num_total_steps, jrandom_key)
        self.H_inv = jnp.linalg.inv(F.f2(x_init)) 
        # self.H_inv = jnp.eye(self.dim)
        self.X_prev = None
        self.grad_curr = None
        self.grad_getter = grad_getter

    def step_getter(self, X, jrandom_key):
        # print(jnp.linalg.eig(jnp.linalg.inv(self.H_inv))[0])
        # print(jnp.linalg.eig(self.F.f2(X))[0])
        # print(self.H_inv)
        # print(jnp.linalg.inv(self.F.f2(X)))
        # print(self.F.f1(X))
        # print(self.grad_curr)
        # print("H MSE", jnp.linalg.norm(self.F.f2(X) - jnp.linalg.inv(self.H_inv), "fro") / jnp.linalg.norm(self.F.f2(X), "fro"))
        # if self.grad_curr is not None:
        #     print("MSE", jnp.linalg.norm(self.F.f1(X) - self.grad_curr) / jnp.linalg.norm(self.F.f1(X)))
        # print("+++++")
        
        self.X_prev = X.copy()
        if self.grad_curr is None:
            self.grad_curr = self.grad_getter(self.F, X, jrandom_key, H=jnp.linalg.inv(self.H_inv))
        f1 = self.grad_curr

        return -self.H_inv.dot(f1), f1
    
    def post_step(self, X, jrandom_key):
        prev_grad = self.grad_curr

        self.grad_curr = self.grad_getter(self.F, X, jrandom_key, H=jnp.linalg.inv(self.H_inv))
        self.H_inv = self.BFGS_update(self.X_prev, X, prev_grad, self.grad_curr, self.H_inv) 
        # self.H_inv = jnp.linalg.inv(self.F.f2(X))
        # self.grad_curr = self.grad_getter(self.F, X, jrandom_key, H=jnp.linalg.inv(self.H_inv))

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

        # print(g_1)
        # print(self.F.f1(x_1))
        # print("x1", repr(x_1))
        # if jnp.linalg.norm(grad_diff) > 50:
        #     with open("./tmp.pkl", "wb") as f:
        #         pickle.dump(jnp.linalg.inv(self.H_inv), f)
        
        ys = jnp.inner(grad_diff, update_step)
        Hy = jnp.dot(H, grad_diff)
        yHy = jnp.inner(grad_diff, Hy)

        # print("Update norm", jnp.linalg.norm(update_step))
        # print("Grad diff norm", jnp.linalg.norm(grad_diff))
        # print("ys", ys)
        # print("yHy norm", jnp.linalg.norm(yHy))
        # print()
        
        H += (ys + yHy) * jnp.outer(update_step, update_step) / ys ** 2
        H -= (jnp.outer(Hy, update_step) + jnp.outer(update_step, Hy)) / ys
        # if ys < 0.1:
        #     H = jnp.eye(x_0.shape[0])

        H = (H + H.T) / 2.
        return H    


class NewtonMethod(OptimizationBlueprint):
    def __init__(self, x_init, F, c1, c2, num_total_steps, jrandom_key):
        super().__init__(x_init, F, c1, c2, num_total_steps, jrandom_key)
    
    def step_getter(self, X, jrandom_key=None):
        f1 = -self.F.f1(X)
        f2 = self.F.f2(X)
        return jnp.linalg.inv(f2).dot(f1), f1


def helper_linesearch(F, c1, c2):

    def helper(x_0, f1_x_0, search_direction, jrandom_key):
        f0 = F.f(x_0)
        dg = jnp.inner(search_direction, f1_x_0)

        def armijo_rule(alpha, jrandom_key):
            return F.f(x_0 + alpha * search_direction, jrandom_key) > f0 + c1*alpha*dg
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        jrandom_key, subkey = jrandom.split(jrandom_key)
        while armijo_rule(alpha, subkey):
            jrandom_key, subkey = jrandom.split(jrandom_key)
            alpha = armijo_update(alpha)

        return alpha

    return helper

def simplex_gradient(F, x_0, S, jrandom_key_f):
    jrandom_key, subkey = jrandom.split(jrandom_key_f)
    FS = F.f(S.T + x_0, subkey)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = F.f(x_0.reshape(1, -1), subkey)
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return SS_inv.dot(S.dot(FS - F_x_0))


def U_FD(h):
    def helper(F, X, jrandom_key, H=None):
        x_0 = X
        if len(x_0.shape) != 1:
            x_0 = x_0.reshape(-1)
        S = jnp.eye(x_0.shape[0])
        return simplex_gradient(F, x_0, h*S, jrandom_key)
    return helper

def USD(max_steps, sig):

    def helper(F, X, jrandom_key, H=None):
        x_0 = X
        if len(x_0.shape) != 1:
            x_0 = x_0.reshape(-1)
        S, _ = optimize_uncentered_S(H, sig, max_steps=max_steps)
        return simplex_gradient(F, x_0, S, jrandom_key)

    return helper


dim = 50
sig = 0.001
jrandom_key = jrandom.PRNGKey(0)


F = Brown(sig)

jrandom_key, subkey = jrandom.split(jrandom_key)
x_init = jrandom.normal(subkey, shape=(dim,)) * 0.2
# x_init = jnp.array([ 0.0097646,   0.00084249,  0.00116959, -0.01149224, -0.01218822])


num_trials = 1
num_steps = 10

c1 = 0.01
c2 = 0.8


# opt = BFGS(x_init, F, c1, c2, num_steps, jrandom_key, grad_getter=lambda F, X, jrandom_key, H=None: F.f1(X))
# plt.plot(jnp.array(opt.run_opt(record_vals=True)[1])[:, 0], label="True")

res = []
for _ in range(num_trials):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    opt = BFGS(x_init, F, c1, c2, num_steps, subkey, grad_getter=USD(10, sig))
    res.append(jnp.array(opt.run_opt(record_vals=True)[1])[:, 0])
# res = jnp.array(res)
# mean_l = jnp.mean(res, axis=0)
# std_l = jnp.std(res, axis=0)
# plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="{} steps USG".format(10))


# res = []
# h = 0.02
# for _ in range(num_trials):
#     jrandom_key, subkey = jrandom.split(jrandom_key)
#     opt = BFGS(x_init, F, c1, c2, num_steps, subkey, grad_getter=U_FD(h))
#     res.append(jnp.array(opt.run_opt(record_vals=True)[1])[:, 0])
# res = jnp.array(res)
# mean_l = jnp.mean(res, axis=0)
# std_l = jnp.std(res, axis=0)
# plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="{} U_FD".format(h))


# # Hyperparamter testing

# # for h in jnp.logspace(-2, 0, 10):
# #     jrandom_key, subkey = jrandom.split(jrandom_key)
# #     opt = BFGS(x_init, F, c1, c2, num_steps, subkey, grad_getter=U_FD(h))
# #     plt.plot(jnp.array(opt.run_opt(record_vals=True)[1])[:, 0], label=str(h))


# plt.yscale("log")
# plt.legend()
# plt.show()


