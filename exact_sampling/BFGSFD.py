import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from simplex_gradient import simplex_gradient

from jax.config import config
config.update("jax_enable_x64", True)


class BFGSFD:
    def __init__(self, sig):
        self.sig = sig

    def grad(self, f, x_0, jrandom_key, H=None):
        dim = len(x_0)
        
        D_diag, U_H = jnp.linalg.eigh(H)
        S = jnp.diag(2 * jnp.sqrt(self.sig / jnp.abs(D_diag)))
        S = U_H.dot(S)

        grad, total_func_calls = simplex_gradient(f, x_0, S, jrandom_key)
        return grad, total_func_calls

        
    
