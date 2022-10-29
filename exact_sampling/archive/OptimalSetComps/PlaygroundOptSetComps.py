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
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")
from Functions import PyCutestGetter
from Ours.generate_sing_vals_V import loss_getter
from simplex_gradient import simplex_gradient

import pow_sampling_set_comps
import U_opt_sampling_set_comps
import opt_FD_comps
import GD_sampling_set_comps

import pickle

from utils import set_loss

verbose = True

pow_res = []
U_opt_res = []
FD_res = []
GD_res = []
F_names = []

seed = 1

test_problem_iter = range(3)


sig = 0.1 

coeff = .1

jrandom_key = jrandom.PRNGKey(seed)

num_mc_runs = 50

for i in tqdm(test_problem_iter):
    F_name, x_0, F = PyCutestGetter(i, sig=sig, noise_type="uniform")
    if verbose:
        print(F_name)
    if F is None:
        continue
    F_names.append(F_name)

    jrandom_key, subkey = jrandom.split(jrandom_key)

    H = F.f2(x_0)

    start_time = time.time()
    l = set_loss(F, x_0, H, sig, coeff, num_mc_runs)
    pow_res.append(pow_sampling_set_comps.get_sampling_set_loss(H, sig, coeff, subkey, l))
    print("Pow time", time.time() - start_time)

    start_time = time.time()   
    l = set_loss(F, x_0, H, sig, coeff, num_mc_runs)
    FD_res.append(opt_FD_comps.get_sampling_set_loss(H, sig, coeff, subkey, l))
    print("FD Time", time.time() - start_time)
    print()
