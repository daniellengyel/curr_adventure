
import jax.numpy as jnp
import jax.random as jrandom
from matplotlib import use
import matplotlib.pyplot as plt

from archive.optimize_USG_error_old import optimize_uncentered_S
from Functions import Quadratic, Ackley, Brown

from tqdm import tqdm



dim = 50
sig = 0.5
jrandom_key = jrandom.PRNGKey(0)


# F = Ackley(sig)
jrandom_key, subkey = jrandom.split(jrandom_key)
Q = jnp.diag(jnp.abs(jrandom.normal(subkey, shape=(dim,))))
F = Quadratic(Q, jnp.zeros(dim), sig)
F = Brown(sig)

jrandom_key, subkey = jrandom.split(jrandom_key)
x_init = jrandom.normal(subkey, shape=(dim,)) * 0.2

num_trials = 5
num_steps = 25

# print("Finite Differences.")
# jrandom_key, subkey = jrandom.split(jrandom_key)
# res = grad_descent(F, x_init, sig, num_trials, num_steps, subkey, max_steps_USG=0, use_exact_gradient=False, use_finite_differences=True)
# mean_l = jnp.mean(res, axis=0)
# std_l = jnp.std(res, axis=0)
# plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="Finite Differences")

# print("Exact Gradient.")
# jrandom_key, subkey = jrandom.split(jrandom_key)
# res = grad_descent(F, x_init, sig, num_trials, num_steps, subkey, max_steps_USG=0, use_exact_gradient=True)
# mean_l = jnp.mean(res, axis=0)
# std_l = jnp.std(res, axis=0)
# plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="Exact Gradient")

# print("0 Steps USG.")
# jrandom_key, subkey = jrandom.split(jrandom_key)
# res = grad_descent(F, x_init, sig, num_trials, num_steps, subkey, max_steps_USG=0, use_exact_gradient=False)
# mean_l = jnp.mean(res, axis=0)
# std_l = jnp.std(res, axis=0)
# plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="0 steps USG")

# max_steps_USG = 10
# print("{} Steps USG.".format(max_steps_USG))
# jrandom_key, subkey = jrandom.split(jrandom_key)
# res = grad_descent(F, x_init, sig, num_trials, num_steps, subkey, max_steps_USG=max_steps_USG, use_exact_gradient=False)
# mean_l = jnp.mean(res, axis=0)
# std_l = jnp.std(res, axis=0)
# plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="{} steps USG".format(max_steps_USG))


dim = 50
sig = 0.01

F = Brown(sig)

num_trials = 5
num_steps = 25

max_steps_USG = 10
print("{} Steps USG.".format(max_steps_USG))
jrandom_key, subkey = jrandom.split(jrandom_key)
res = grad_descent(F, x_init, sig, num_trials, num_steps, subkey, max_steps_USG=max_steps_USG, use_exact_gradient=False)
mean_l = jnp.mean(res, axis=0)
std_l = jnp.std(res, axis=0)
plt.errorbar(range(len(mean_l)), mean_l, std_l, fmt='o', markersize=8, capsize=20, label="{} steps USG with 0.01 noise".format(max_steps_USG))

plt.legend()
plt.show()





