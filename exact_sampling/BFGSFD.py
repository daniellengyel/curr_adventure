import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jax.config import config
config.update("jax_enable_x64", True)


class BFGSFD:
    def __init__(self, sig):
        self.sig = sig

    def grad(self, f, x_0, jrandom_key, H=None):
        dim = len(x_0)

        D, U = jnp.linalg.eigh(H)

        all_dirs = jnp.eye(dim)
        grad = np.zeros(dim)
        total_func_calls = 0
        jrandom_key, subkey = jrandom.split(jrandom_key)
        f_x_0 = f.f(x_0, subkey)
        total_func_calls += 1
        for i in range(dim):
            curr_dir = all_dirs[i]
            jrandom_key, subkey = jrandom.split(jrandom_key)
            h = 2 * jnp.sqrt(self.sig/jnp.abs(D[i]))
            grad[i] = (f.f(x_0 + h * curr_dir) - f_x_0)/h
            total_func_calls += 1
        return grad, total_func_calls
    
