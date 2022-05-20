import jax.numpy as jnp
import jax.random as jrandom

from simplex_gradient import simplex_gradient


class FD:
    def __init__(self, sig, is_central, h=0.1):
        self.sig = sig
        self.h = h
        self.is_central = is_central

    def grad(self, F, X, jrandom_key, H=None):
        x_0 = X
        if len(x_0.shape) != 1:
            x_0 = x_0.reshape(-1)
       
        if self.is_central:
            S = jnp.eye(len(x_0)) * self.h
            S = jnp.concatenate([S, -S], axis=1)
        else:
            S = jnp.eye(len(x_0)) * self.h
        return simplex_gradient(F, x_0, S, jrandom_key)
        