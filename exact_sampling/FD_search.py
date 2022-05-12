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
        
        # if self.best_h is None:
        #     jrandom_key, subkey = jrandom.split(jrandom_key)
            # self.best_h = get_best_S(F, x_0, subkey, lb=-3, up=1, num_trials=50, num_MC_trials=50)[0, 0]
       
        if self.is_central:
            S = jnp.eye(len(x_0)) * self.h
            S = jnp.concatenate([S, -S], axis=1)
        else:
            S = jnp.eye(len(x_0)) * self.h
        return simplex_gradient(F, x_0, S, jrandom_key)

def get_mc_loss(F, x_0, num_runs):
    g_F = F.f1(x_0)
    def helper(S, jrandom_key):
        errs = []
        for _ in range(num_runs):
            jrandom_key, subkey = jrandom.split(jrandom_key)
            sgd = simplex_gradient(F, x_0, S, subkey)[0]
            errs.append(float(jnp.linalg.norm(sgd - g_F)**2))
        errs = jnp.array(errs)
        return jnp.mean(errs)
    return helper


def get_best_S(F, x_0, jrandom_key, lb=-3, up=1, num_trials=50, num_MC_trials=50):
    l = get_mc_loss(F, x_0, num_MC_trials)
    dim = x_0.shape[0]
    curr_best = None
    curr_best_val = float("inf")
    S = jnp.eye(dim)
    for h in jnp.logspace(lb, up, num_trials):
        jrandom_key, subkey = jrandom.split(jrandom_key)
        curr_err = l(S * h, subkey)
        if curr_err < curr_best_val:
            curr_best_val = curr_err
            curr_best = h

    return curr_best * S
