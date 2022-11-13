import multiprocessing
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import math

import time
from multiprocessing import Pool
import multiprocessing

from simplex_gradient import simplex_gradient

import sys
sys.path.append("/Users/daniellengyel/curr_adventure/exact_sampling/Ours_utils")
from pow_sampling_set import create_approx_S


class Ours:
    def __init__(self, sig, max_h=0):
        self.sig = sig
        self.max_h = max_h

        self.last_S = None
        self.last_X = None
        self.last_H = None
        self.last_F = None

    def grad(self, F, X, jrandom_key, H):
        x_0 = X
        if (self.last_F == F) and jnp.all(self.last_X == X) and jnp.all(self.last_H == H):
            S = self.last_S
        else:
            if len(x_0.shape) != 1:
                x_0 = x_0.reshape(-1)

            if hasattr(F, "_get_dists"):
                curr_max_h = min(self.max_h, min(F._get_dists(X)[0]))
            else:
                curr_max_h = self.max_h

            S = create_approx_S(H, self.sig, curr_max_h)
            self.last_S = S

            self.last_X = X
            self.last_H = H
            self.last_F = F

        return simplex_gradient(F, x_0, S, jrandom_key)