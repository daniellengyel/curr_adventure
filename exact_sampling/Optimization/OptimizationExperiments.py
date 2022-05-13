import jax.numpy as jnp
import jax.random as jrandom
from Optimization import BFGS, GradientDescent, NewtonMethod
from pdfo import newuoa



from save_load import save_opt
from tqdm import tqdm

import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")
from pow_sampling_set import pow_SG

from Functions import PyCutestGetter
from AdaptiveFD import adapt_FD
from BFGSFD import BFGSFD 
from FD_search import FD
from NEWUO_test import NEWUOA_Wrapper

from multiprocessing import Pool
import multiprocessing

NUM_CPU = 1

def run_our_GD(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose, coeff, max_h):
    print(seed)
    jrandom_key = jrandom.PRNGKey(seed=seed)
    
    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)

    sig = F.sig
    noise_type = F.noise_type

    grad_getter = pow_SG(sig, coeff=coeff, max_h=max_h)
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    final_X, our_res, our_res_X = optimizer.run_opt()
    save_opt(our_res, our_res_X, "OurMethod_GD", F_name, sig, noise_type, c1, c2, seed, {"coeff":coeff, "max_h":max_h})


def run_adaptive_GD(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose):
    print(seed)
    jrandom_key = jrandom.PRNGKey(seed=seed)
    
    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)

    sig = F.sig
    noise_type = F.noise_type

    grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    final_X, adaptFD_res, adaptFD_res_X = optimizer.run_opt()
    save_opt(adaptFD_res, adaptFD_res_X, "AdaptFD", F_name, sig, noise_type, c1, c2, seed)


def run_FD_GD(F_name, x_0, sig, noise_type, grad_eps, h, c1, c2, num_total_steps, seed, verbose, is_central):
    print(seed)
    jrandom_key = jrandom.PRNGKey(seed=seed)
    
    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)

    sig = F.sig
    noise_type = F.noise_type
    grad_getter = FD(sig, is_central=is_central, h=h) 
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    final_X, FD_res, FD_res_X = optimizer.run_opt()
    if is_central:
        save_name = "CentralFD_GD"
    else:
        save_name = "FD_GD"

    save_opt(FD_res, FD_res_X, save_name, F_name, sig, noise_type, c1, c2, seed, {"h": h})


if __name__ == "__main__":
    sig = 1
    noise_type="uniform"

    c1 = 0.1
    c2 = 0.9
    num_total_steps = 50
    grad_eps = 1e-5
    seed = 0

    jrandom_key = jrandom.PRNGKey(seed)

    verbose = False

    test_problem_iter = range(2, 80)
    dim_iter = range(10)

    num_trials = 50

    # # Newtons Method
    # print("++++++ Newtons Method ++++++")
    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         jrandom_key = jrandom.PRNGKey(seed=seed)

    #         optimizer = NewtonMethod(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose=verbose)
    #         final_X, exact_res, exact_res_X = optimizer.run_opt()
    #         save_opt(exact_res, exact_res_X, "NewtonsMethod", F_name, sig, noise_type, c1, c2, seed)

    # # Gradient Descent
    # print("++++++ Gradient Descent ++++++")
    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break

    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         jrandom_key = jrandom.PRNGKey(seed=seed)

    #         optimizer = GradientDescent(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_eps, verbose=verbose)
    #         final_X, exact_res, exact_res_X = optimizer.run_opt()
    #         save_opt(exact_res, exact_res_X, "GradientDescent", F_name, sig, noise_type, c1, c2, seed)


    # # adaptive FD
    # print("++++++ Adaptive FD ++++++")
    # pool = Pool(processes=NUM_CPU)

    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         inp_list = [(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose) for seed in range(num_trials)]
    #         pool.starmap(run_AdaptFD_GD, inp_list)

    # pool.close()

    # Forward FD
    print("++++++ FD ++++++")
    h = 0.1
    # pool = Pool(processes=NUM_CPU)

    for f_i in tqdm(test_problem_iter):
        for d_i in tqdm(dim_iter):
            F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
            if F is None:
                break
            
            print("F Name", F_name)
            print("F Dim", len(x_0))

            is_central = False
            inp_list = [(F_name, x_0, sig, noise_type, grad_eps, h, c1, c2, num_total_steps, seed, verbose, is_central) for seed in range(num_trials)]
            # pool.starmap(run_FD_GD, inp_list)
            for inp in inp_list:
                run_FD_GD(*inp)

    # pool.close()


    # Central FD
    print("++++++ FD ++++++")
    h = 0.5
    pool = Pool(processes=NUM_CPU)

    for f_i in tqdm(test_problem_iter):
        for d_i in tqdm(dim_iter):
            F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
            if F is None:
                break
            
            print("F Name", F_name)
            print("F Dim", len(x_0))

            is_central = True
            inp_list = [(F_name, x_0, sig, noise_type, grad_eps, h, c1, c2, num_total_steps, seed, verbose, is_central) for seed in range(num_trials)]
            pool.starmap(run_FD_GD, inp_list)
            
    pool.close()


    # # Our Method
    # print("++++++ Our Method ++++++")
    # pool = Pool(processes=NUM_CPU)
    # coeff = 0
    # max_h = 2

    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         inp_list = [(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose, coeff, max_h) for seed in range(num_trials)]
    #         pool.starmap(run_our_GD, inp_list)
    
    # pool.close()

        
    # # NEWUOA
    # print("++++++ NEWUOA ++++++")
    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         jrandom_key = jrandom.PRNGKey(seed=seed)

    #         curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
    #         curr_F = lambda X: float(curr_F_inst.f(X))

    #         newuoa_full_res = newuoa(curr_F, x_0) 
    #         newuoa_res = newuoa_full_res["fhist"]
    #         save_opt(newuoa_res, "NEWUOA", F_name, sig, noise_type, c1, c2, seed)
