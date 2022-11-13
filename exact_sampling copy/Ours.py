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

        # all_pow_U = generate_all_pow_U(len(S_pow_index_set))

    def grad(self, F, X, jrandom_key, H):

        x_0 = X
        if len(x_0.shape) != 1:
            x_0 = x_0.reshape(-1)

        if hasattr(F, "_get_dists"):
            curr_max_h = min(self.max_h, min(F._get_dists(X)[0]))
        else:
            curr_max_h = self.max_h


        S = create_approx_S(H, self.sig, curr_max_h)
        # print(S)
        return simplex_gradient(F, x_0, S, jrandom_key)