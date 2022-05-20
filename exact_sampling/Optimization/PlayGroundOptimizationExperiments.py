import jax.numpy as jnp
import jax.random as jrandom
from save_load import save_opt
from Optimization import BFGS, NewtonMethod
from pdfo import newuoa

from NEWUO_test import NEWUOA_Wrapper

import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")
from pow_sampling_set import pow_SG
from Functions import PyCutestGetter
from AdaptiveFD import adapt_FD
from FD import FD

from tqdm import tqdm

import matplotlib.pyplot as plt


if __name__ == "__main__":
    sig = 5
    noise_type="uniform"

    step_size = 0.001
    num_total_steps = 50
    grad_eps = 1e-5
    seed = 0

    jrandom_key = jrandom.PRNGKey(seed)

    test_problem_iter = range(38, 39)
    dim_iter = range(10)

    # # regular BFGS
    # for i in test_problem_iter:
    #     print(i)
    #     for d in dim_iter:
    #         F_name, x_0, F = PyCutestGetter(func_i=i, dim_i = d, sig=sig, noise_type=noise_type)
    #         if F is None:
    #             break
    #         print("F name", F_name)
    #         print("dim", len(x_0))
    #         print()
            # optimizer = NewtonMethod(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_eps, verbose=True)
            # final_X, exact_res = optimizer.run_opt()


    # adaptive FD
    # test_problem_iter = [2] # [19] # range(0, 1)
    for i in tqdm(test_problem_iter):
        F_name, x_0, F = PyCutestGetter(func_i=i, dim_i=0, sig=sig, noise_type=noise_type)
        print(F_name)
        grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
        optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
        final_X, adaptFD_res, _ = optimizer.run_opt()
        # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", step_size, seed)

    # standard FD
    # test_problem_iter = [2] # [19] # range(0, 1)
    for i in tqdm(test_problem_iter):
        F_name, x_0, F = PyCutestGetter(func_i=i, dim_i=0, sig=sig, noise_type=noise_type)
        print(F_name)
        grad_getter = FD(sig, is_central=False, h=0.1) 
        optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
        final_X, FD_res, _ = optimizer.run_opt()
        # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", step_size, seed)

    # Central FD
    # test_problem_iter = [2] # [19] # range(0, 1)
    for i in tqdm(test_problem_iter):
        F_name, x_0, F = PyCutestGetter(func_i=i, dim_i=0, sig=sig, noise_type=noise_type)
        print(F_name)
        grad_getter = FD(sig, is_central=True, h=0.1) 
        optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
        final_X, central_FD_res, _ = optimizer.run_opt()
        # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", step_size, seed)


    # Our Method
    # test_problem_iter = [2] # range(15, 54) # range(0, 1)
    for i in tqdm(test_problem_iter):
        F_name, x_0, F = PyCutestGetter(func_i=i, dim_i=0, sig=sig, noise_type=noise_type)
        
        if F is None:
            continue
        print(F_name)

        grad_getter = pow_SG(sig, coeff=0, max_h=0.1)
        optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
        final_X, our_res, _ = optimizer.run_opt()
        # _, our_res = save_opt(opt_res, "OurMethod", F_name, sig, noise_type, step_size, seed)

    # # NEWUOA
    # for i in tqdm(test_problem_iter):
    #     F_name, x_0, F = PyCutestGetter(i, sig=sig, noise_type=noise_type)
        
    #     if F is None:
    #         continue
    #     print(F_name)


    #     curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
    #     curr_F = lambda X: float(curr_F_inst.f(X))


    #     newuoa_full_res = newuoa(curr_F, x_0) 
    #     newuoa_res = newuoa_full_res["fhist"]
    #     # _, our_res = save_opt(opt_res, "OurMethod", F_name, sig, noise_type, step_size, seed)


    # plt.plot(adaptFD_res[:, 1], adaptFD_res[:, 0], label="Adapt")
    # plt.plot(our_res[:, 1], our_res[:, 0], label="Our")
    # plt.xlabel("Time")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    plt.plot(adaptFD_res[:, 2], adaptFD_res[:, -1], label="Adapt")
    plt.plot(FD_res[:, 2], FD_res[:, -1], label="FD")
    plt.plot(central_FD_res[:, 2], central_FD_res[:, -1], label="Central FD")
    plt.plot(our_res[:, 2], our_res[:, -1], label="Our")
    # # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Func Calls")
    plt.yscale("log")
    plt.legend()
    plt.show()

    # plt.plot(exact_res[:, 2], exact_res[:, 0], label="Exact")
    plt.plot(adaptFD_res[:, 2], adaptFD_res[:, 0], label="Adapt")
    plt.plot(FD_res[:, 2], FD_res[:, 0], label="FD")
    plt.plot(central_FD_res[:, 2], central_FD_res[:, 0], label="Central FD")
    plt.plot(our_res[:, 2], our_res[:, 0], label="Our")
    # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Func Calls")
    plt.yscale("log")
    plt.legend()
    plt.show()

    # plt.plot(range(len(exact_res[:, 2])), exact_res[:, 0], label="Exact")
    plt.plot(range(len(adaptFD_res[:, 2])), adaptFD_res[:, 0], label="Adapt")
    plt.plot(range(len(FD_res[:, 2])), FD_res[:, 0], label="FD")
    plt.plot(range(len(central_FD_res[:, 2])), central_FD_res[:, 0], label="Central FD")
    plt.plot(range(len(our_res[:, 2])), our_res[:, 0], label="Our")
    # # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Num Iter")
    plt.yscale("log")
    plt.legend()
    plt.show()



