import jax.numpy as jnp
import jax.random as jrandom

from simplex_gradient import simplex_gradient


class FD:
    def __init__(self, sig, is_central, h=0.1, use_H=False):
        self.sig = sig
        self.h = h
        self.is_central = is_central
        self.use_H = use_H

        self.last_S = None
        self.last_X = None
        self.last_H = None
        self.last_F = None

    def grad(self, F, X, jrandom_key, H=None):
        x_0 = X
        if (self.last_F == F) and jnp.all(self.last_X == X) and jnp.all(self.last_H == H):
            S = self.last_S
        else:
            if len(x_0.shape) != 1:
                x_0 = x_0.reshape(-1)
        
            if self.is_central:
                S = jnp.eye(len(x_0)) * self.h
                S = jnp.concatenate([S, -S], axis=1)
            else:
                if self.use_H and (H is not None):
                    S = jnp.diag(2 * jnp.sqrt(self.sig / jnp.abs(jnp.diag(H))))
                    if self.h is not None:
                        S = jnp.minimum(S, self.h)
                else:
                    S = jnp.eye(len(x_0)) * self.h

            # print(S)
            self.last_S = S
            self.last_X = X
            self.last_H = H
            self.last_F = F
    
        return simplex_gradient(F, x_0, S, jrandom_key)
        