
import jax.numpy as jnp
import jax.random as jrandom
from matplotlib import use
import matplotlib.pyplot as plt

from archive.optimize_USG_error_old import optimize_uncentered_S
from Functions import Quadratic, Ackley, Brown

from tqdm import tqdm

def simplex_gradient(F, x_0, S, subkey_f):
    jrandom_key, subkey = jrandom.split(subkey_f)
    FS = F.f(S.T + x_0, subkey)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = F.f(x_0.reshape(1, -1), subkey)
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return SS_inv.dot(S.dot(FS - F_x_0))

def helper_linesearch(f, c1, c2):

    def helper(X, search_direction):
        f0 = f(X)
        f1 = -search_direction
        dg = jnp.inner(search_direction, f1)

        def armijo_rule(alpha):
            return f(X + alpha * search_direction) > f0 + c1*alpha*dg
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        while armijo_rule(alpha):
            alpha = armijo_update(alpha)

        return alpha

    return helper


def grad_descent(F, x_init, sig, num_trials, num_steps, jrandom_key, max_steps_USG, use_exact_gradient, use_finite_differences=False):
    res = []
    if use_finite_differences:
        h = 0.5
        S = h * jnp.eye(x_init.shape[0])
    linesearch = helper_linesearch(F.f, c1=0.01, c2=0.7)
    for _ in range(num_trials):
        x_curr = x_init
        curr_res = []
        for t in tqdm(range(num_steps)):
            # curr_res.append(F.f(x_curr))
            jrandom_key, subkey = jrandom.split(jrandom_key)
            if not use_exact_gradient:
                if not use_finite_differences:
                    H = F.f2(x_curr)
                    # H_err = jrandom.normal(subkey, H.shape) * 0.1 # TODO with noise. 
                    # print(jnp.linalg.norm(H_err.T.dot(H_err), "fro")/jnp.linalg.norm(H, "fro"))
                    # H += H_err.T.dot(H_err)
                    jrandom_key, subkey = jrandom.split(jrandom_key)
                    S, _ = optimize_uncentered_S(H, sig, max_steps=max_steps_USG)
                sg = simplex_gradient(F, x_curr, S, subkey)
                curr_res.append(jnp.linalg.norm(sg - F.f1(x_curr))/jnp.linalg.norm(F.f1(x_curr)))
            else:
                sg = F.f1(x_curr)

            search_direction = -sg

            alpha = linesearch(x_curr, search_direction)
            x_curr += alpha*search_direction    

        res.append(curr_res)
        if use_exact_gradient:
            break

    res = jnp.array(res)
    return res



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





