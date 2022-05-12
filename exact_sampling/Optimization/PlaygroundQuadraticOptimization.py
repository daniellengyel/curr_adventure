import jax.numpy as jnp
import jax.random as jrandom
from save_load import save_opt
from Optimization import BFGS, NewtonMethod
from pdfo import newuoa

from NEWUO_test import NEWUOA_Wrapper

import sys 
sys.path.append("..")
from pow_sampling_set import pow_SG
from Functions import PyCutestGetter, Quadratic
from AdaptiveFD import adapt_FD
from FD_search import FD
from BFGSFD import BFGSFD

from tqdm import tqdm

import matplotlib.pyplot as plt


if __name__ == "__main__":
    sig = 2
    noise_type="uniform"

    c1 = 0.1
    c2 = 0.9
    num_total_steps = 20
    grad_eps = 1e-5
    seed = 1

    jrandom_key = jrandom.PRNGKey(seed)

    dim = 32
    Q = jnp.diag(jnp.logspace(-2, 4, dim))
    b = jnp.zeros(dim)
    F = Quadratic(Q, b, sig=sig, noise_type="uniform")

    x_0 = jnp.ones(dim)

    # regular BFGS
    optimizer = NewtonMethod(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose=True)
    final_X, exact_res = optimizer.run_opt()



    # adaptive FD
    # # test_problem_iter = [2] # [19] # range(0, 1)
    # for i in tqdm(test_problem_iter):
    #     F_name, x_0, F = PyCutestGetter(i, sig=sig, noise_type=noise_type)
    #     print(F_name)
    #     grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
    #     optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
    #     final_X, adaptFD_res = optimizer.run_opt()
    #     # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", c1, c2, seed)

    # standard FD
    # grad_getter = FD(sig, is_central=False) 
    # grad_getter = BFGSFD(sig) 
    FD_res_total = []
    for _ in range(3):
        grad_getter = FD(sig, is_central=False) 
        jrandom_key, subkey = jrandom.split(jrandom_key)
        optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, subkey, grad_getter, grad_eps, verbose=True)
        final_X, FD_res = optimizer.run_opt()
        FD_res_total.append(FD_res[:, 2]) 
    # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", c1, c2, seed)
    FD_res_total = jnp.array(FD_res_total)
    FD_res_mean = jnp.mean(FD_res_total, axis=0)
    FD_res_std = jnp.std(FD_res_total, axis=0)


    # Central FD
    grad_getter = FD(sig, is_central=True) 
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
    final_X, central_FD_res = optimizer.run_opt()
    # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", c1, c2, seed)


    # Our Method
    grad_getter = pow_SG(sig, coeff=0)
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
    final_X, our_res = optimizer.run_opt()
    # _, our_res = save_opt(opt_res, "OurMethod", F_name, sig, noise_type, c1, c2, seed)

    # # NEWUOA
    # curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
    # curr_F = lambda X: float(curr_F_inst.f(X))
    # newuoa_full_res = newuoa(curr_F, x_0) 
    # newuoa_res = newuoa_full_res["fhist"]
    # # _, our_res = save_opt(opt_res, "OurMethod", F_name, sig, noise_type, c1, c2, seed)


    # plt.plot(adaptFD_res[:, 1], adaptFD_res[:, 0], label="Adapt")
    # plt.plot(our_res[:, 1], our_res[:, 0], label="Our")
    # plt.xlabel("Time")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    # # plt.plot(adaptFD_res[:, 2], adaptFD_res[:, -1], label="Adapt")
    # plt.plot(FD_res[:, 2], FD_res[:, -1], label="FD")
    # plt.plot(central_FD_res[:, 2], central_FD_res[:, -1], label="Central FD")
    # plt.plot(our_res[:, 2], our_res[:, -1], label="Our")
    # # plt.plot(newuoa_res, label="NEWUOA")
    # plt.xlabel("Func Calls")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    # # plt.plot(adaptFD_res[:, 2], adaptFD_res[:, 0], label="Adapt")
    # plt.plot(exact_res[:, 2], exact_res[:, 0], label="Exact")
    # plt.plot(FD_res[:, 2], FD_res[:, 0], label="FD")
    # plt.plot(central_FD_res[:, 2], central_FD_res[:, 0], label="Central FD")
    # plt.plot(our_res[:, 2], our_res[:, 0], label="Our")
    # # plt.plot(newuoa_res, label="NEWUOA")
    # plt.xlabel("Func Calls")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()


    # plt.plot(adaptFD_res[:, 2], adaptFD_res[:, 0], label="Adapt")
    plt.plot(range(len(exact_res[:, 2])), exact_res[:, 0], label="Exact")
    # plt.plot(range(len(FD_res[:, 2])), FD_res[:, 0], label="FD")
    plt.errorbar(range(len(FD_res_mean)), FD_res_mean, yerr=FD_res_std, label="FD")
    plt.plot(range(len(central_FD_res[:, 2])), central_FD_res[:, 0], label="Central FD")
    plt.plot(range(len(our_res[:, 2])), our_res[:, 0], label="Our")
    # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Number Iterations")
    plt.yscale("log")
    plt.legend()
    plt.show()



