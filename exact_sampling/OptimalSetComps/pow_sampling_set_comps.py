
import jax.numpy as jnp
import jax.random as jrandom

from jax.config import config
config.update("jax_enable_x64", True)

import sys 
sys.path.append("../")
from pow_sampling_set import create_approx_S


def get_sampling_set_loss(H, sig, coeff, jrandom_key, loss):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    S = create_approx_S(H, sig, coeff, subkey)

    return [loss(S, jrandom_key)]