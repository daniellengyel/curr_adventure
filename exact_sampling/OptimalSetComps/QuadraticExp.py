import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax import grad

from tqdm import tqdm
import time

from jax.config import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")
from Functions import Quadratic

import pow_sampling_set_comps
import U_opt_sampling_set_comps
import opt_FD_comps
import FD_search_comps
import GD_sampling_set_comps


import pickle

from utils import set_loss, post_process


pow_res = []
U_opt_res = []
FD_res = []
GD_res = []
F_names = []

seed = 1

sig = 0.1 

coeff = 0.1

jrandom_key = jrandom.PRNGKey(seed)

dim = 4

jrandom_key, subkey = jrandom.split(jrandom_key)

x_0 = jnp.ones(dim)

N = 20

pow_res = []
FD_search_res = []

for _ in tqdm(range(N)):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    Q = jrandom.normal(subkey, shape=(dim, dim,))
    Q = Q + Q.T #@ Q.T #(H + H.T)/2.
    F = Quadratic(Q, jnp.zeros(dim), sig, noise_type="gaussian")
    H = F.f2(x_0)
    l = set_loss(F, x_0, H, sig, 0, num_runs=100)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    pow_curr_res = pow_sampling_set_comps.get_sampling_set_loss(H, sig, coeff, subkey, l)
    FD_search_curr_res = GD_sampling_set_comps.optimize_uncentered_S(H, sig, coeff, subkey, l, max_steps=25)

    # FD_search_curr_res = opt_FD_comps.get_sampling_set_loss(H, sig, coeff, subkey, l)

    # FD_search_curr_res = FD_search_comps.get_sampling_set_loss(F, x_0, subkey, l)

    pow_res.append(pow_curr_res[-1])
    FD_search_res.append(FD_search_curr_res[-1])


_, _, pow_mc_loss, _, _ = post_process(pow_res)
_, _, FD_search_mc_loss, _, _ = post_process(FD_search_res)


plt.scatter(pow_mc_loss, FD_search_mc_loss)

plt.plot(jnp.linspace(0, max(max(pow_mc_loss), max(FD_search_mc_loss)), 100), jnp.linspace(0, max(max(pow_mc_loss), max(FD_search_mc_loss)), 100))

plt.xlim(0, 0.001 + max(max(pow_mc_loss), max(FD_search_mc_loss)))
plt.ylim(0, 0.001 + max(max(pow_mc_loss), max(FD_search_mc_loss)))

plt.show()



