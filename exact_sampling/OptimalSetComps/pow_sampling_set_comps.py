
import jax.numpy as jnp
import jax.random as jrandom

from jax.config import config
config.update("jax_enable_x64", True)

import time

import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")
from pow_sampling_set import create_approx_S


def get_sampling_set_loss(H, sig, coeff, jrandom_key, loss):
    S = create_approx_S(H, sig, coeff)
    return [loss(S, jrandom_key)]