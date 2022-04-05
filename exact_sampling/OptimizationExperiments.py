import jax.numpy as jnp
import jax.random as jrandom
from Functions import PyCutestIterator
from Optimization import BFGS, NewtonMethod
from AdaptiveFD import adapt_FD
from save_load import save_opt
from USimplex import USD
from tqdm import tqdm

import matplotlib.pyplot as plt

eps = 0.1
noise_type="uniform"

# num_test_problems = 14
test_problem_iter = [1] # range(0, 1)
test_problem_iter = PyCutestIterator(test_problem_iter, eps=eps, noise_type=noise_type)

if noise_type == "uniform":
    sig = eps / jnp.sqrt(3)

c1 = 0.1
c2 = 0.9
num_total_steps = 50
grad_eps = 1e-5
seed = 0

jrandom_key = jrandom.PRNGKey(seed)

# # regulad BFGS
# test_problem_iter = [1] # range(0, 1)
# test_problem_iter = PyCutestIterator(test_problem_iter, eps=0, noise_type=noise_type)
# for F_name, x_0, F in test_problem_iter:
#     print(F_name)
#     optimizer = NewtonMethod(x_0, F, c1, c2, num_total_steps, jrandom_key, grad_eps)
#     final_X, exact_res = optimizer.run_opt()
#     # save_opt(opt_res, "NewtonExact", F_name, sig, "uniform", c1, c2, seed)
# plt.plot(exact_res[:, 1], exact_res[:, 0], label="Newton")
# plt.plot(exact_res[:, 2], exact_res[:, 0], label="Newton")

verbose = False

# adaptive FD
test_problem_iter = None # range(0, 1)
test_problem_iter = PyCutestIterator(test_problem_iter, eps=eps, noise_type=noise_type)
for F_name, x_0, F in tqdm(test_problem_iter):
    if verbose:
        print(F_name)
    grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, jrandom_key, grad_getter, grad_eps, verbose)
    final_X, adaptFD_res = optimizer.run_opt()
    save_opt(adaptFD_res, "AdaptFD", F_name, eps, "uniform", c1, c2, seed)

verbose = False

test_problem_iter = None # range(0, 1)
test_problem_iter = PyCutestIterator(test_problem_iter, eps=eps, noise_type=noise_type)
# Our Method
max_steps = 100
for F_name, x_0, F in tqdm(test_problem_iter):
    if verbose:
        print(F_name)
    grad_getter = USD(max_steps, sig)
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, jrandom_key, grad_getter, grad_eps, verbose)
    final_X, our_res = optimizer.run_opt()
    our_res = save_opt(our_res, "OurMethod", F_name, eps, noise_type, c1, c2, seed)



