import jax.numpy as jnp
import jax.random as jrandom
from save_load import save_opt
from Optimization import BFGS, NewtonMethod, Trust, GradientDescent
from pdfo import newuoa

from NEWUO_test import NEWUOA_Wrapper

import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")

from pow_sampling_set import pow_SG
from Functions import PyCutestGetter, Quadratic, generate_quadratic
from AdaptiveFD import adapt_FD
from FD import FD
from BFGSFD import BFGSFD

from tqdm import tqdm

import matplotlib.pyplot as plt


if __name__ == "__main__":
    dim = 10
    test_problem_iter = ["{}_{}_{}_{}_{}".format(dim, "log", -2, 4, 0)]
    x_0 = jnp.zeros(dim)

    sig = 0
    step_size = 1e-5
    noise_type="uniform"

    num_total_steps = 10
    grad_eps = 1e-5

    verbose = True

    F_no_noise = generate_quadratic(test_problem_iter[0], 0, noise_type)
    F = generate_quadratic(test_problem_iter[0], sig, noise_type)


    print(jnp.linalg.eigh(F.f2(x_0))[0])
    print(F.b)

    jrandom_key = jrandom.PRNGKey(0)

    # # regular BFGS
    # optimizer = NewtonMethod(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose=True)
    # final_X, exact_res = optimizer.run_opt()


    GD_sig = 0
    optimizer = GradientDescent(x_0, F_no_noise, step_size, num_total_steps, GD_sig, None, grad_eps, verbose=True)
    final_X, exact_res, _ = optimizer.run_opt()

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

    # grad_getter = FD(sig, is_central=False, h=1.) 
    # optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    # final_X, FD_res, _ = optimizer.run_opt()
    # # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", c1, c2, seed)



    # # Central FD
    # grad_getter = FD(sig, is_central=True, h=1.) 
    # optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    # final_X, central_FD_res, _ = optimizer.run_opt()
    # # save_opt(adaptFD_res, "AdaptFD", F_name, sig, "uniform", c1, c2, seed)


    # # Our Method
    # grad_getter = pow_SG(sig, max_h=1.)
    # optimizer = Trust(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    # final_X, our_res, _ = optimizer.run_opt()
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

    plt.plot(range(len(FD_res[:, 2])), FD_res[:, 0], label="FD")
    plt.plot(range(len(central_FD_res[:, 2])), central_FD_res[:, 0], label="Central FD")
    plt.plot(range(len(our_res[:, 2])), our_res[:, 0], label="Our")
    # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Number Iterations")
    plt.yscale("log")
    plt.legend()
    plt.show()



