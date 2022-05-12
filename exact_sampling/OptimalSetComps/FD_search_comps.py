import jax.numpy as jnp
import jax.random as jrandom
import sys
sys.path.append("..")

from FD_search import get_best_S

def get_sampling_set_loss(F, x_0, jrandom_key, loss):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    S = get_best_S(F, x_0, jrandom_key=subkey, lb=-3, up=1, num_trials=10, num_MC_trials=50)
    return [loss(S, jrandom_key)]

