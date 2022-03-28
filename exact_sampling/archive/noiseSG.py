import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import pandas as pd

from Functions import Quadratic, Ackley 
from optimize_USG_error import optimize_uncentered_S


from scipy.stats import linregress

from tqdm import tqdm

from jax.config import config
config.update("jax_enable_x64", True)

def noise_simplex_gradient(S, subkey_f, sig):
    jrandom_key, subkey = jrandom.split(subkey_f) 
    FS = jrandom.normal(subkey, shape=(S.shape[1], )) * sig
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = jrandom.normal(subkey, shape=(1, )) * sig
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return SS_inv.dot(S.dot(FS - F_x_0))


dim = 5
seed = 0
jrandom_key = jrandom.PRNGKey(seed)


N = dim
hs = jnp.logspace(-2, 1, 10)

jrandom_key, subkey = jrandom.split(jrandom_key)

S = jrandom.normal(subkey, shape=(dim, N))
SS_inv = jnp.linalg.inv(S.dot(S.T))



sig = 0.1

num_trials = 100000
res = []
for _ in tqdm(range(num_trials)):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    res.append(jnp.linalg.norm(noise_simplex_gradient(S, subkey, sig))**2)


res = jnp.array(res)
print("Empirical Mean", jnp.mean(res))
print("Emperical Std", jnp.std(res)/jnp.sqrt(num_trials))

first_term = sig**2 * jnp.linalg.norm(SS_inv.dot(S), ord="fro")**2 
print(first_term)
S_sum = jnp.sum(S,axis=1)
second_term_con = sig**2 * S_sum.T.dot(SS_inv.dot(SS_inv.dot(S_sum)))
second_term = sig**2 * jnp.sum(jnp.array([si.T.dot(SS_inv.dot(SS_inv.dot(jnp.sum(S,axis=1)))) for si in S.T]))
print(first_term + second_term)
print("con", first_term + second_term_con)