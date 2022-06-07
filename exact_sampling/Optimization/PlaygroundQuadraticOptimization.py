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

from pow_sampling_set import pow_SG
from Functions import HeartDisease, load_cutest_quadratic, generate_quadratic
from AdaptiveFD import adapt_FD
from FD import FD
from ExactGrad import ExactGrad


from tqdm import tqdm

import matplotlib.pyplot as plt


if __name__ == "__main__":

    sig = 10
    step_size = 5e-5
    noise_type="uniform"
    h = 50

    num_total_steps = 100
    grad_eps = 1e-4

    verbose = True

    dim = 32

    test_problem_iter = ["{}_{}_{}_{}_{}".format(dim, "lin", 0.001, 1000, 0)]
    F_no_noise, x_0 = generate_quadratic(test_problem_iter[0], 0, noise_type)
    F, x_0 = generate_quadratic(test_problem_iter[0], sig, noise_type)

    # F = HeartDisease(sig, noise_type)
    # F_no_noise = HeartDisease(0, noise_type)
    # dim = len(F.opt_X)
    # x_0 = jnp.ones(dim)/jnp.sqrt(dim)

    # lmbda = 10
    # F = load_cutest_quadratic("DUAL4", lmbda, sig, noise_type)
    # F_no_noise = load_cutest_quadratic("DUAL4", lmbda, sig, noise_type)
    # x_0 = F.x0

    jrandom_key = jrandom.PRNGKey(1)



    GD_sig = 0
    grad_getter = ExactGrad()
    optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  
    final_X, exact_res, _ = optimizer.run_opt()

    # # adaptive FD
    # # # test_problem_iter = [2] # [19] # range(0, 1)
    # # for i in tqdm(test_problem_iter):
    # #     F_name, x_0, F = PyCutestGetter(i, sig=sig, noise_type=noise_type)
    # #     print(F_name)
    # #     grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
    # #     optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=True)
    # #     final_X, adaptFD_res = optimizer.run_opt()

    # standard FD
    grad_getter = FD(sig, is_central=False, h=h, use_H=True) 
    optimizer = InterpH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose, smoothing=1)  
    final_X, FD_res, _ = optimizer.run_opt()


    # # Central FD
    # grad_getter = FD(sig, is_central=True, h=h) 
    # optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose, smoothing=20)  
    # final_X, central_FD_res, _ = optimizer.run_opt()


    # # Our Method
    grad_getter = pow_SG(sig, max_h=h)
    optimizer = InterpH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose, smoothing=1)  
    final_X, our_res, _ = optimizer.run_opt()

    # # NEWUOA
    # curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
    # curr_F = lambda X: float(curr_F_inst.f(X))
    # newuoa_full_res = newuoa(curr_F, x_0) 
    # newuoa_res = newuoa_full_res["fhist"]


    # plt.plot(adaptFD_res[:, 1], adaptFD_res[:, 0], label="Adapt")
    # plt.plot(our_res[:, 1], our_res[:, 0], label="Our")
    # plt.xlabel("Time")
    # plt.yscale("log")
    # plt.legend()
    # plt.show()

    # # plt.plot(adaptFD_res[:, 2], adaptFD_res[:, -1], label="Adapt")
    plt.plot(FD_res[:, 2], FD_res[:, -1], label="FD")
    # plt.plot(central_FD_res[:, 2], central_FD_res[:, -1], label="Central FD")
    plt.plot(our_res[:, 2], our_res[:, -1], label="Our")
    # # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Func Calls")
    # plt.yscale("log")
    plt.legend()
    plt.show()

    # # plt.plot(adaptFD_res[:, 2], adaptFD_res[:, 0], label="Adapt")
    plt.plot(exact_res[:, 2], exact_res[:, 0], label="Exact")
    plt.plot(FD_res[:, 2], FD_res[:, 0], label="FD")
    # plt.plot(central_FD_res[:, 2], central_FD_res[:, 0], label="Central FD")
    plt.plot(our_res[:, 2], our_res[:, 0], label="Our")
    # # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Func Calls")
    plt.yscale("log")
    plt.legend()
    plt.show()


    # plt.plot(adaptFD_res[:, 2], adaptFD_res[:, 0], label="Adapt")
    plt.plot(range(len(exact_res[:, 2])), exact_res[:, 0], label="Exact")
    # plt.plot(range(len(FD_res[:, 2])), FD_res[:, 0], label="FD")

    plt.plot(range(len(FD_res[:, 2])), FD_res[:, 0], label="FD")
    # plt.plot(range(len(central_FD_res[:, 2])), central_FD_res[:, 0], label="Central FD")
    plt.plot(range(len(our_res[:, 2])), our_res[:, 0], label="Our")
    # plt.plot(newuoa_res, label="NEWUOA")
    plt.xlabel("Number Iterations")
    plt.yscale("log")
    plt.legend()
    plt.show()



