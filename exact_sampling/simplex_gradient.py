import jax.numpy as jnp
from jax import random as jrandom

def simplex_gradient(F, x_0, S, jrandom_key_f):
    num_func_calls = 0
    jrandom_key, subkey = jrandom.split(jrandom_key_f)
    FS = []
    for s_i in S.T:
        jrandom_key, subkey = jrandom.split(jrandom_key)
        FS.append(F.f(s_i + x_0, subkey))
        num_func_calls += 1
    FS = jnp.array(FS)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = F.f(x_0, subkey)
    num_func_calls += 1
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return SS_inv.dot(S.dot(FS - F_x_0)), num_func_calls


