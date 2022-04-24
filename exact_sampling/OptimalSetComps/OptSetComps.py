import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax import grad


import scipy 

import matplotlib.pyplot as plt


from tqdm import tqdm
import time

from jax.config import config
config.update("jax_enable_x64", True)


import sys 
sys.path.append("../")
from Functions import PyCutestGetter
from generate_sing_vals_V import loss_getter
from simplex_gradient import simplex_gradient

import pow_sampling_set_comps
import U_opt_sampling_set_comps

def opt_set_loss(F, x_0, H, sig, coeff):
    l = loss_getter(len(x_0), H, sig, coeff)
    g_F = F.f1(x_0)
    start_time = time.time()
    def helper(S, jrandom_key):
        sgd = simplex_gradient(F, x_0, S, jrandom_key)[0]
        return [time.time() - start_time, l(S.T.flatten()), jnp.linalg.norm(sgd - g_F)**2]
    return helper

verbose = True

pow_res = []
U_opt_res = []

seed = 0

test_problem_iter = range(3, 8)

sig = 0.1
coeff = 0.1

jrandom_key = jrandom.PRNGKey(seed)

for i in tqdm(test_problem_iter):
    F_name, x_0, F = PyCutestGetter(i, eps=0, noise_type=None)
    if verbose:
        print(F_name)
    if F is None:
        continue

    jrandom_key, subkey = jrandom.split(jrandom_key)

    H = F.f2(x_0)

    l = opt_set_loss(F, x_0, H, sig, coeff)
    pow_res.append(jnp.array(pow_sampling_set_comps.get_sampling_set_loss(H, sig, coeff, subkey, l)))

    l = opt_set_loss(F, x_0, H, sig, coeff)
    U_opt_res.append(jnp.array(U_opt_sampling_set_comps.get_sampling_set_loss(H, sig, coeff, subkey, l, num_iter=10)))


    print(pow_res)
    print(U_opt_res)

    plt.scatter(pow_res[-1][:, 0], pow_res[-1][:, 1], label="pow loss bound")
    plt.scatter(pow_res[-1][:, 0], pow_res[-1][:, 2], label="pow loss MC")

    plt.plot(U_opt_res[-1][:, 0], U_opt_res[-1][:, 1], label="U_opt loss bound")
    plt.plot(U_opt_res[-1][:, 0], U_opt_res[-1][:, 2], label="U_opt loss MC")

    plt.legend()
    plt.show()
