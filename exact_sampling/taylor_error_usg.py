import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import pandas as pd
import os

from Functions import Quadratic, Ackley, Brown
from taylor_error_utils import get_errs

from scipy.stats import linregress

from tqdm import tqdm

from jax.config import config
config.update("jax_enable_x64", True)

plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 24
plt.rcParams["figure.figsize"] = (10, 10)


def main(F, sig, hs, dim, N, jrandom_seed, plot_name, trial_runs, num_sample_sets, fig_save):
    
    x_0 = jnp.array(list(range(1, dim+1)))
    x_0 /= jnp.linalg.norm(x_0)
    
    jrandom_key = jrandom.PRNGKey(jrandom_seed)
    jrandom_key, subkey = jrandom.split(jrandom_key)

    plt.scatter([], [], label="Emperical Error", color="grey")
    plt.plot([], [], label="Analytic Error Bound", color="grey")

    for _ in range(num_sample_sets):
        S = jrandom.normal(subkey, shape=(dim, N))
        S = S / jnp.linalg.norm(S)

        res = None
        for _ in range(trial_runs):
            jrandom_key, subkey = jrandom.split(jrandom_key)
            if res is None:
                res = get_errs(F, x_0, S, hs, subkey)
            else:
                res += get_errs(F, x_0, S, hs, subkey)
        
        res /= trial_runs

        plt.scatter(res["h"], res["err"])
        plt.plot(res["h"], res["tight_quadratic_bound"] + res["noise_bound"])

    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Normalized MSE")
    plt.xlabel("h")
    plt.legend()
    plt.title(plot_name)
    if fig_save:
        path = "./figs/taylor_error_usg"
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + "/" + plot_name + ".png", dpi=330)
    
    plt.close()

    # plt.show()

sample_runs = 3
num_sample_sets = 3

# ==== 25 dim, 0.001 sig =====
dim = 25
sig = 0.001
N = dim
jrandom_seed = 0
fig_save = True

# # ++++ Ackley +++++
F = Ackley(sig)

hs = jnp.logspace(-1, 2, 25)
fig_name = "Ackley_sig-{}_dim-{}".format(sig, dim)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, num_sample_sets, fig_save)

# ++++ Quadratic ++++
jrandom_key_quad = jrandom.PRNGKey(0)
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
Q = jrandom.normal(subkey, shape=(dim, dim,))
Q = jnp.diag(jnp.linspace(0.5, 100, dim))
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
b = jrandom.normal(subkey, shape=(dim,))
F = Quadratic(Q, b, sig)

hs = jnp.logspace(-2, 1, 25)
fig_name = "Quadratic_sig-{}_dim-{}".format(sig, dim)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, num_sample_sets, fig_save)

# ==== 25 dim, 0.01 sig =====
dim = 25
sig = 0.01
N = dim
jrandom_seed = 0
fig_save = True

# # ++++ Ackley +++++
F = Ackley(sig)

hs = jnp.logspace(-1, 2, 25)
fig_name = "Ackley_sig-{}_dim-{}".format(sig, dim)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, num_sample_sets, fig_save)

# ++++ Quadratic ++++
jrandom_key_quad = jrandom.PRNGKey(0)
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
Q = jrandom.normal(subkey, shape=(dim, dim,))
Q = jnp.diag(jnp.linspace(0.5, 100, dim))
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
b = jrandom.normal(subkey, shape=(dim,))
F = Quadratic(Q, b, sig)

hs = jnp.logspace(-2, 1, 25)
fig_name = "Quadratic_sig-{}_dim-{}".format(sig, dim)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, num_sample_sets, fig_save)

# ==== 25 dim, 0.1 sig =====
dim = 25
sig = 0.1
N = dim
jrandom_seed = 3
fig_save = True

# # ++++ Ackley +++++
F = Ackley(sig)

hs = jnp.logspace(-1, 2, 25)
fig_name = "Ackley_sig-{}_dim-{}".format(sig, dim)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, num_sample_sets, fig_save)

# ++++ Quadratic ++++
jrandom_key_quad = jrandom.PRNGKey(0)
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
Q = jrandom.normal(subkey, shape=(dim, dim,))
Q = jnp.diag(jnp.linspace(0.5, 100, dim))
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
b = jrandom.normal(subkey, shape=(dim,))
F = Quadratic(Q, b, sig)

hs = jnp.logspace(-2, 1, 25)
fig_name = "Quadratic_sig-{}_dim-{}".format(sig, dim)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, num_sample_sets, fig_save)



