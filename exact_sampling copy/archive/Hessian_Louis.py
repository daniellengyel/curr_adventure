import re
import jax.numpy as jnp
import jax.random as jrandom
from Functions import Quadratic

import matplotlib.pyplot as plt

from jax import jit, grad, jacfwd

from tqdm import tqdm 

plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 24
plt.rcParams["figure.figsize"] = (10, 10)

from jax.config import config
config.update("jax_enable_x64", True)

jrandom_key = jrandom.PRNGKey(0)
jrandom_key, subkey = jrandom.split(jrandom_key)

d = 10
A = jrandom.normal(subkey, shape=(d, d,))
F = Quadratic(A, jnp.ones(d))

x_0 = jnp.ones(d) / jnp.linalg.norm(jnp.ones(d))

H = F.f2(x_0)
res = []
for N_float in tqdm(jnp.logspace(1, 3, 10)):
    N = int(N_float)
    jrandom_key, subkey = jrandom.split(jrandom_key)

    X = jrandom.normal(subkey, shape=(d, N))
    X = (X.T - jnp.mean(X, axis=1)).T * 0.1

    F_mean = jnp.mean(jnp.array([F.f(x + x_0) for x in X.T]))
    cov = jnp.cov(X, bias=True)
    cov_inv = jnp.linalg.inv(cov)
    fXX = jnp.mean(jnp.array([F.f(x + x_0) * jnp.outer(x, x) for x in X.T]), axis=0)
    curr_est = cov_inv.dot(fXX - F_mean * cov).dot(cov_inv)
    res.append(jnp.linalg.norm(curr_est - H)/jnp.linalg.norm(H))

print(curr_est)
print(H)
plt.plot(jnp.logspace(1, 3, 10), res)
plt.ylabel("|H - H_est|/|H|")
plt.xlabel("N")
plt.yscale("log")
plt.xscale("log")
plt.show()