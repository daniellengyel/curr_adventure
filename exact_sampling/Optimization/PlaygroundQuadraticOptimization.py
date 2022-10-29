import jax.numpy as jnp
import jax.random as jrandom
from save_load import save_opt
from Optimization import ExactH_GD, InterpH_GD
from pdfo import newuoa

from NEWUO_test import NEWUOA_Wrapper

from jax.config import config
config.update("jax_enable_x64", True)

import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")

from Ours import Ours
from Functions import HeartDisease, load_cutest_quadratic, generate_quadratic, PyCutestGetter
from archive.AdaptiveFD import adapt_FD
from FD import FD
from ExactGrad import ExactGrad


from tqdm import tqdm

import matplotlib.pyplot as plt


if __name__ == "__main__":

    sig = 20
    step_size = 5e-5
    noise_type="uniform"
    h = 2
    smoothing = 200

    num_total_steps = 300
    grad_eps = 1e-4

    verbose = True

    # dim = 16

    # test_problem_iter = ["{}_{}_{}_{}_{}".format(dim, "lin", 0.001, 1000, 0)]
    # test_problem_iter = ["{}_{}_{}_{}_{}".format(dim, "log", -1, 2, 1)]
    func_dim = 100
    func_name = "BDQRTIC"
    func_name, x_0, F_no_noise = PyCutestGetter(func_name=func_name, func_dim=func_dim, sig=0, noise_type=noise_type) # generate_quadratic(test_problem_iter[0], 0, noise_type)
    func_name, x_0, F = PyCutestGetter(func_name=func_name, func_dim=func_dim, sig=sig, noise_type=noise_type) # generate_quadratic(test_problem_iter[0], sig, noise_type)

    jrandom_key = jrandom.PRNGKey(1)

    run_interp = True

    print("GD")
    GD_sig = 0
    grad_getter = ExactGrad()
    optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    final_X, exact_res, _ = optimizer.run_opt()

    print("Exact FD")
    # standard FD
    grad_getter = FD(sig, is_central=False, h=h, use_H=True) 
    optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    final_X, Exact_FD_res, _ = optimizer.run_opt()

    if run_interp:
        print("Interp FD")
        grad_getter = FD(sig, is_central=False, h=0.1, use_H=False) 
        optimizer = InterpH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose, smoothing=smoothing)  
        final_X, Interp_FD_res, _ = optimizer.run_opt()

    # if run_interp:
    #     print("Interp FD")
    #     grad_getter = FD(sig, is_central=False, h=1, use_H=False) 
    #     optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    #     final_X, Interp_FD_res, _ = optimizer.run_opt()


    print("Exact Our")
    # # Our Method
    grad_getter = Ours(sig, max_h=h)
    optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    final_X, Exact_our_res, _ = optimizer.run_opt()

    if run_interp:
        print("Interp Our")
        grad_getter = Ours(sig, max_h=h)
        optimizer = InterpH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose, smoothing=smoothing)  
        final_X, Interp_our_res, _ = optimizer.run_opt()

    
    if run_interp:
        plt.plot(range(len(Interp_our_res[:, 2])), Interp_our_res[:, -1], label="Interp Our")
        plt.plot(range(len(Interp_FD_res[:, 2])), Interp_FD_res[:, -1], label="Interp FD")

    plt.plot(range(len(Exact_FD_res[:, 2])), Exact_FD_res[:, -1], label="Exact FD")
    plt.plot(range(len(Exact_our_res[:, 2])), Exact_our_res[:, -1], label="Exact Our")
    plt.xlabel("Number Iterations")
    # plt.yscale("log")
    plt.legend()
    plt.show()


    if run_interp:
        plt.plot(range(len(Interp_FD_res[:, 2])), Interp_FD_res[:, 0], label="Interp FD")
        plt.plot(range(len(Interp_our_res[:, 2])), Interp_our_res[:, 0], label="Interp Our")

    F_opt = 0 # F.f(F.x_opt)
    plt.plot(range(len(exact_res[:, 2])), exact_res[:, 0] - F_opt, label="Exact")
    plt.plot(range(len(Exact_FD_res[:, 2])), Exact_FD_res[:, 0] - F_opt, label="Exact FD")
    plt.plot(range(len(Exact_our_res[:, 2])), Exact_our_res[:, 0] - F_opt, label="Exact Our")
    plt.xlabel("Number Iterations")
    plt.yscale("log")
    plt.legend()
    plt.show()



