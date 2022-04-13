import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import pandas as pd

from Functions import Quadratic, Ackley, Brown
from archive.optimize_USG_error_old import optimize_uncentered_S


from scipy.stats import linregress

from tqdm import tqdm

from jax.config import config
config.update("jax_enable_x64", True)

def simplex_gradient(F, x_0, S, subkey_f):
    jrandom_key, subkey = jrandom.split(subkey_f)
    FS = F.f(S.T + x_0, subkey)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = F.f(x_0.reshape(1, -1), subkey)
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return SS_inv.dot(S.dot(FS - F_x_0))

def tight_quad_bound(H, S):
    s_vec = jnp.sum(jnp.array([S[:, i] * S[:, i].T.dot(H.dot(S[:, i])) for i in range(S.shape[1])]), axis=0)
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return 1/2. * jnp.linalg.norm(SS_inv.dot(s_vec))

def noise_bound(sig, S):
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    S_sum = jnp.sum(S, axis=1)
    return jnp.sqrt(sig**2 * jnp.linalg.norm(SS_inv.dot(S), ord="fro")**2 + sig**2 * S_sum.dot(SS_inv.dot(SS_inv.dot(S_sum))))


def get_errs(F, x_0, S, hs, jrandom_key):
    res = []
    B = jnp.max(jnp.abs(F.f2(x_0.reshape(1, -1))[0]))


    for h in tqdm(hs):
        curr_S = h * S
        jrandom_key, subkey_f = jrandom.split(jrandom_key)
        sg = simplex_gradient(F, x_0, curr_S, subkey_f)
        eig_min_squared = min(jnp.linalg.eig(curr_S.dot(curr_S.T))[0]).real
        R = max(jnp.linalg.norm(curr_S, axis=0))

        res.append({
            "eig_min": eig_min_squared,
            "R": R, 
            "h": h,
            "eig_min_squared": eig_min_squared,
            "H_bound": B,
            "err": jnp.linalg.norm(sg - F.f1(x_0))**2 / jnp.linalg.norm(F.f1(x_0))**2,
            "tight_quadratic_bound": tight_quad_bound(F.f2(x_0), curr_S)**2 / jnp.linalg.norm(F.f1(x_0))**2,
            "noise_bound": noise_bound(F.sig, curr_S)**2 / jnp.linalg.norm(F.f1(x_0))**2,
        })


    return pd.DataFrame(res, dtype=float)

