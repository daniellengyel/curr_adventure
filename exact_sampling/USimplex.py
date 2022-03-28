import jax.numpy as jnp
from jax import random as jrandom
from optimize_USG_error import optimize_uncentered_S

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



class USD:
    def __init__(self, max_steps, sig):
        self.max_steps = max_steps
        self.sig = sig
        self.prev_rot_S = None

    def grad(self, F, X, jrandom_key, H=None):
        x_0 = X
        if len(x_0.shape) != 1:
            x_0 = x_0.reshape(-1)
        assert len(x_0) == 4
        
        S, _, pre_rot_S = optimize_uncentered_S(H, self.sig, self.max_steps, self.prev_rot_S)
        # self.pre_rot_S = pre_rot_S

        return simplex_gradient(F, x_0, S, jrandom_key)

