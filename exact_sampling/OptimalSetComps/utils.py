import jax.numpy as jnp
import jax.random as jrandom
import pandas as pd

from tqdm import tqdm

import time

from jax.config import config
config.update("jax_enable_x64", True)

from Ours.generate_sing_vals_V import loss_getter
from simplex_gradient import simplex_gradient

def set_loss(F, x_0, H, sig, coeff, num_runs):
    l = loss_getter(len(x_0), H, sig, coeff)
    g_F = F.f1(x_0)
    start_time = time.time()
    def helper(S, jrandom_key):
        errs = []
        for _ in range(num_runs):
            jrandom_key, subkey = jrandom.split(jrandom_key)
            sgd = simplex_gradient(F, x_0, S, subkey)[0]
            errs.append(float(jnp.linalg.norm(sgd - g_F)**2))
        errs = jnp.array(errs)
        return (jnp.array([time.time() - start_time, l(S.T.flatten()), jnp.mean(errs), jnp.linalg.norm(g_F)]), errs)
    return helper

def post_process(data):
    times = []
    analytic_loss = []
    mc_loss = []
    min_mc_loss = []
    max_mc_loss = []
    for i in range(len(data)):
        times.append(data[i][0][0])
        analytic_loss.append(data[i][0][1])
        mc_loss.append(jnp.mean(data[i][1]))
        min_mc_loss.append(jnp.percentile(data[i][1], 0.25))
        max_mc_loss.append(jnp.percentile(data[i][1], 0.75))
        
    return times, analytic_loss, mc_loss, min_mc_loss, max_mc_loss

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

