import pandas as pd
import numpy as np

from jax import random as jrandom
import jax.numpy as jnp
import jax
from jax import jit, partial, grad, jacfwd

from tqdm import tqdm 
import itertools

def best_width_search(f, upper_bound, lower_bound, num_samples, search_type="log"):
    
    if search_type == "log":
        search_space = jnp.logspace(jnp.log(lower_bound)/jnp.log(10), jnp.log(upper_bound)/jnp.log(10), num_samples)
    else:
        search_space = jnp.linspace(lower_bound, upper_bound, num_samples)
        
    f_vals = [f(x) for x in search_space]
    curr_best = float("inf")
    curr_h = 0
    for i in range(len(search_space)):
        if abs(curr_best) > abs(f_vals[i]):
            curr_best = f_vals[i]
            curr_h = search_space[i]
            
    return curr_h, curr_best
    
def hessian_comp(true_H, approx_H_func, true_grad=None):    
    def helper(h):
        approx_H = approx_H_func(h)
        
        if true_grad is None:
            return jnp.mean(approx_H - true_H)
        
        return newton_dir_loss(true_H, approx_H, true_grad)
    
    return helper

def mse_loss(true_H, approx_H):
    """MSE of: approx and true"""
    return [float(jnp.mean((approx_H - true_H)**2))]

def noisy_mse_loss(true_H, approx_H, noisy_approx_H):
    """MSE of: Approx, Approx with Noise, Noise"""
    return [float(jnp.mean((approx_H - true_H)**2)), float(jnp.mean((noisy_approx_H - true_H)**2)), float(jnp.mean((noisy_approx_H - approx_H)**2))]

def newton_dir_loss(true_H, approx_H, true_grad):
    # cosine similarity 

    trueHg = jnp.linalg.inv(true_H).dot(true_grad)
    trueHg /= jnp.linalg.norm(trueHg)
    approxHg = jnp.linalg.inv(approx_H).dot(true_grad)
    approxHg /= jnp.linalg.norm(approxHg)

    return 1 - approxHg.dot(trueHg)

from itertools import product

class LossHistory:
    
    def __init__(self, conf_names):
        self.conf_names = conf_names
        self.d = {}
        
    def generate_key(self, truth_vals):
        res = ""
        for i in range(len(truth_vals)):
            res += self.conf_names[i] + ":" + str(truth_vals[i]) + "_"
        return res[:-1]
    
    def add_val(self, truth_vals, val):
        curr_key = self.generate_key(truth_vals)
        if curr_key not in self.d:
            self.d[curr_key] = []
        self.d[curr_key].append(val)
        
    def plot(self):
        for k, v in self.d.items():
            plt.plot(jnp.array(v)[:, 1], label=k)
            
        plt.legend()
        plt.show()
        
    def hist(self):
        for k, v in self.d.items():
            plt.hist(jnp.array(v)[:, 1], label=k)
            plt.legend()
            plt.show()

class Logger():

    def __init__(self, hp_names, loss_names=["loss_approx"]):
        self.hp_names = hp_names
        self.d = pd.DataFrame(columns=hp_names + loss_names)

    def add_value(self, hp_vals, loss_val):
        """hp_vals needs to have the hyperparameter values in the order as given by hp_names"""
        self.d = self.d.append(pd.Series(hp_vals + loss_val, index=self.d.columns), ignore_index=True)
    

    def subset(self, hp_vals):
        f = np.ones(len(self.d))

        for i, hp_v in enumerate(hp_vals):
            k = self.d.columns[i]
            if hp_v is None:
                continue
            
            f *= 1*(self.d[k] == hp_v)

        return self.d[f == 1]

    def unique_combs(self):
        return self.d.drop_duplicate(subset = ["function", "dimension", "std", "frac_samples"])

    def mean_std(self):
        return self.d.groupby(["function", "dimension", "std", "frac_samples"]).mean(), self.d.groupby(["function", "dimension", "std", "frac_samples"]).std()



def run_tests(estimator, funcs, x_0_generator, standard_deviations, dimensions, frac_samples, num_runs, seed, loss_type="Newton_Dir"):
    logger = Logger(["function", "dimension", "std", "frac_samples", "jrandom_key"])

    jrandom_key = jrandom.PRNGKey(seed)

    for _ in tqdm(range(num_runs)):
        
        for dim in dimensions:
            for func_name in funcs:
                F = funcs[func_name](dim)
                x_0 = x_0_generator(dim)
                true_H = F.f2(x_0.reshape(1, -1))[0]
                true_G = F.f1(x_0.reshape(1, -1))[0]
    
                for std, frac_s in itertools.product(standard_deviations, frac_samples): 
                    num_samples = int(frac_s*dim)

                    jrandom_key, subkey = jrandom.split(jrandom_key)
                    
                    est_G = estimator(F, x_0, std, num_samples, subkey)
                    
                    logger.add_value([func_name, dim, std, frac_s, subkey], mse_loss(true_G, est_G))

                    
    return logger


