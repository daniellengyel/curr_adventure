import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import pandas as pd
import os

from Functions import Quadratic, Ackley, Brown
from taylor_error_utils import get_errs
from optimize_USG_error import optimize_uncentered_S

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


def main(F, sig, hs, dim, N, jrandom_seed, plot_name, trial_runs, fig_save, H_err_fracs):
    
    x_0 = jnp.array(list(range(1, dim+1)))
    x_0 /= jnp.linalg.norm(x_0)
    jrandom_key = jrandom.PRNGKey(jrandom_seed)


    num_opt_steps = 15

    for i in range(len(H_err_fracs)):

        H = F.f2(x_0)

        jrandom_key, subkey = jrandom.split(jrandom_key)
        H_delta = jrandom.normal(subkey, shape=H.shape)
        H += H_delta / jnp.linalg.norm(H_delta) * (H_err_fracs[i] * jnp.linalg.norm(H))
        S, opt_loss = optimize_uncentered_S(H, sig, max_steps=num_opt_steps)
        S = S / jnp.linalg.norm(S)

        res = None
        for _ in range(trial_runs):
            jrandom_key, subkey = jrandom.split(jrandom_key)
            if res is None:
                res = get_errs(F, x_0, S, hs, subkey)
            else:
                res += get_errs(F, x_0, S, hs, subkey)
        
        res /= trial_runs

        plt.scatter(res["h"], res["err"], label="H err{} empirical".format(H_err_fracs[i]))
        plt.plot(res["h"], res["tight_quadratic_bound"] + res["noise_bound"], label="H err{} analytic bound".format(H_err_fracs[i]))

    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Normalized MSE")
    plt.xlabel("h")
    plt.legend()
    plt.title(plot_name)
    if fig_save:
        path = "./figs/taylor_error_usg_opt"
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + "/" + plot_name + ".png", dpi=330)
    else:
        plt.show()

    plt.close()


sample_runs = 5

# ==== 25 dim, 0.001 sig, 0.001 H_err =====
dim = 25
sig = 0.001
N = dim
jrandom_seed = 0
fig_save = True
H_err_fracs = [0, 0.1, 0.5, 1, 2]

# # ++++ Ackley +++++
F = Ackley(sig)

hs = jnp.logspace(-1, 2, 25)
fig_name = "Ackley_sig-{}_dim-{}_H_errs-{}".format(sig, dim, H_err_fracs)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, fig_save, H_err_fracs)

# ++++ Quadratic ++++
jrandom_key_quad = jrandom.PRNGKey(0)
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
Q = jrandom.normal(subkey, shape=(dim, dim,))
Q = jnp.diag(jnp.linspace(0.5, 100, dim))
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
b = jrandom.normal(subkey, shape=(dim,))
F = Quadratic(Q, b, sig)

hs = jnp.logspace(-2, 1, 25)
fig_name = "Quadratic_sig-{}_dim-{}_H_errs-{}".format(sig, dim, H_err_fracs)
main(F, sig, hs, dim, N, jrandom_seed, fig_name, sample_runs, fig_save, H_err_fracs)


