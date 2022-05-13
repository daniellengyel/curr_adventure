import jax.numpy as jnp
import jax.random as jrandom
from Optimization import BFGS, GradientDescent, NewtonMethod
from pdfo import newuoa

from NEWUO_test import NEWUOA_Wrapper


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

    test_problem_iter = range(0, 100)
    dim_iter = range(10)

    num_trials = 25

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
    #         final_X, exact_res = optimizer.run_opt()
    #         save_opt(exact_res, "NewtonsMethod", F_name, sig, noise_type, c1, c2, seed)

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
    #         final_X, exact_res = optimizer.run_opt()
    #         save_opt(exact_res, "GradientDescent", F_name, sig, noise_type, c1, c2, seed)


    # adaptive FD
    print("++++++ Adaptive FD ++++++")
    for f_i in tqdm(test_problem_iter):
        for d_i in tqdm(dim_iter):
            F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
            if F is None:
                break
            
            print("F Name", F_name)
            print("F Dim", len(x_0))

            for seed in range(num_trials):
                jrandom_key = jrandom.PRNGKey(seed=seed)

                grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
                optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
                final_X, adaptFD_res = optimizer.run_opt()
                save_opt(adaptFD_res, "AdaptFD", F_name, sig, noise_type, c1, c2, seed)

    # # Forward FD
    # print("++++++ FD ++++++")
    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         jrandom_key = jrandom.PRNGKey(seed=seed)

    #         h = 0.1
    #         grad_getter = FD(sig, is_central=False, h=h) 
    #         optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    #         final_X, FD_res = optimizer.run_opt()
    #         save_opt(FD_res, "FD_GD", F_name, sig, noise_type, c1, c2, seed, {"h": h})


    # # Central FD
    # print("++++++ FD ++++++")
    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         jrandom_key = jrandom.PRNGKey(seed=seed)

    #         h = 0.5
    #         grad_getter = FD(sig, is_central=True, h=h) 
    #         optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    #         final_X, FD_res = optimizer.run_opt()
    #         save_opt(FD_res, "CentralFD_GD", F_name, sig, noise_type, c1, c2, seed, {"h": h})


    # # Our Method
    # print("++++++ Our Method ++++++")
    # for f_i in tqdm(test_problem_iter):
    #     for d_i in tqdm(dim_iter):
    #         F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
            
    #         if F is None:
    #             break
            
    #         print("F Name", F_name)
    #         print("F Dim", len(x_0))

    #         jrandom_key = jrandom.PRNGKey(seed=seed)

    #         coeff = 0
    #         max_h = 2

    #         grad_getter = pow_SG(sig, coeff=coeff, max_h=max_h)
    #         optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    #         final_X, our_res = optimizer.run_opt()
    #         save_opt(our_res, "OurMethod_GD", F_name, sig, noise_type, c1, c2, seed, {"coeff":coeff, "max_h":max_h})

        
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
