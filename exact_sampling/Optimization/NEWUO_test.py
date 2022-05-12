from pdfo import newuoa
# If SciPy (version 1.1 or above) is installed, then Bounds, LinearConstraint,
# and NonlinearConstraint can alternatively be imported from scipy.optimize.

import jax.numpy as jnp
import jax.random as jrandom

from tqdm import tqdm

import sys 
sys.path.append("..")

from Functions import PyCutestGetter

class NEWUOA_Wrapper:
    def __init__(self, func, jrandom_key):
        self.func = func
        self.jrandom_key = jrandom_key

    def f(self, X, is_random=True):
        self.jrandom_key, subkey = jrandom.split(self.jrandom_key)
        if is_random:
            out = self.func.f(jnp.array(X), subkey)
        else:
            out = self.func.f(jnp.array(X), None)
        return out

if __name__ == "__main__":
    jrandom_key = jrandom.PRNGKey(0)

    test_problem_iter = [2]

    for i in tqdm(test_problem_iter):
        F_name, x_0, F = PyCutestGetter(i, sig=0.1, noise_type="uniform")
        
        print(F_name)
        
        if F is None:
            continue

        jrandom_key, subkey = jrandom.split(jrandom_key)
        curr_F_inst = NEWUOA_Wrapper(F, subkey)
        curr_F = lambda X: float(curr_F_inst.f(X))


        res = newuoa(curr_F, x_0)  
        print(len(res["fhist"]))
        print("True val", curr_F_inst.f(res["x"], is_random=False))