import jax.numpy as jnp
import jax.random as jrandom
from Optimization import BFGS, GradientDescent, NewtonMethod
from pdfo import newuoa

from jax.config import config
config.update("jax_enable_x64", True)

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

NUM_CPU = 1 # int(os.getenv("NUM_CPU"))
ARRAY_INDEX = 1 # os.getenv("PBS_ARRAY_INDEX")
if ARRAY_INDEX is None:
    ARRAY_INDEX = 1
else:
    ARRAY_INDEX = int(ARRAY_INDEX)
    
NUM_ARRAY = 1

def run_exp(opt_type, F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose, param_dict={}):
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)

    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)
    
    if opt_type == "NEWUOA":
        curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
        curr_F = lambda X: float(curr_F_inst.f(X))

        newuoa_full_res = newuoa(curr_F, x_0) 
        newuoa_res = newuoa_full_res["fhist"]
        
        
        
    save_opt(newuoa_res, None, "NEWUOA", F_name, dim, sig, noise_type, c1, c2, seed, param_dict)

def run_NEWUOA(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose):
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)

    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)
    curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
    curr_F = lambda X: float(curr_F_inst.f(X))

    newuoa_full_res = newuoa(curr_F, x_0) 
    newuoa_res = newuoa_full_res["fhist"]
    save_opt(newuoa_res, None, "NEWUOA", F_name, dim, sig, noise_type, c1, c2, seed)

def run_our_GD(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose, coeff, max_h, NUM_CPU):
    print(seed)
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)
    
    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)

    sig = F.sig
    noise_type = F.noise_type

    grad_getter = pow_SG(sig, coeff=coeff, max_h=max_h, NUM_CPU=NUM_CPU)
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)    
    final_X, our_res, our_res_X = optimizer.run_opt()
    if grad_getter.pool is not None:
        grad_getter.pool.close()
    save_opt(our_res, our_res_X, "OurMethod_GD", F_name, dim, sig, noise_type, c1, c2, seed, {"coeff":coeff, "max_h":max_h})


def run_adaptive_GD(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose):
    print(seed)
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)
    
    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)

    sig = F.sig
    noise_type = F.noise_type

    grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
    optimizer = BFGS(x_0, F, c1, c2, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)
    final_X, adaptFD_res, adaptFD_res_X = optimizer.run_opt()
    save_opt(adaptFD_res, adaptFD_res_X, "AdaptFD_GD", F_name, dim, sig, noise_type, c1, c2, seed)


def run_FD_GD(F_name, x_0, sig, noise_type, grad_eps, h, c1, c2, num_total_steps, seed, verbose, is_central):
    print(seed)
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)
    
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

    save_opt(FD_res, FD_res_X, save_name, F_name, dim, sig, noise_type, c1, c2, seed, {"h": h})


if __name__ == "__main__":

    OPT_TYPES = ["Our_GD"] # os.getenv("OPT_TYPES").split(" ")

    sig = 20
    noise_type="uniform"

    c1 = 0.1
    c2 = 0.9
    num_total_steps = 50
    grad_eps = 1e-5
    seed = 0

    jrandom_key = jrandom.PRNGKey(seed)

    verbose = True

    test_problem_iter = range(11, 12)
    dim_iter = range(0, 10)

    num_trials = 50
    lower_seed, upper_seed = int(num_trials / NUM_ARRAY) * (ARRAY_INDEX - 1), min(int(num_trials / NUM_ARRAY) * (ARRAY_INDEX ), num_trials)


    # Newtons Method
    newton_sig = 0
    if "Newton" in OPT_TYPES:
        print("++++++ Newtons Method ++++++")
        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=newton_sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                jrandom_key = jrandom.PRNGKey(seed=seed)

                optimizer = NewtonMethod(x_0, F, c1, c2, num_total_steps, newton_sig, jrandom_key, grad_eps, verbose=verbose)
                final_X, exact_res, exact_res_X = optimizer.run_opt()
                save_opt(exact_res, exact_res_X, "NewtonsMethod", F_name, len(x_0), newton_sig, noise_type, c1, c2, seed)

    # Gradient Descent
    GD_sig = 0
    if "GD" in OPT_TYPES:
        print("++++++ Gradient Descent ++++++")
        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=0, noise_type=noise_type)
                
                if F is None:
                    break

                print("F Name", F_name)
                print("F Dim", len(x_0))

                jrandom_key = jrandom.PRNGKey(seed=seed)

                optimizer = GradientDescent(x_0, F, c1, c2, num_total_steps, GD_sig, jrandom_key, grad_eps, verbose=verbose)
                final_X, exact_res, exact_res_X = optimizer.run_opt()
                save_opt(exact_res, exact_res_X, "GradientDescent", F_name, len(x_0), GD_sig, noise_type, c1, c2, seed)


    # adaptive FD
    if "AdaptFD_GD" in OPT_TYPES:
        print("++++++ Adaptive FD ++++++")

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                inp_list = [(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_adaptive_GD(*inp)


    # Forward FD
    if "FD_GD" in OPT_TYPES:
        print("++++++ FD ++++++")
        h = 0.5

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                is_central = False
                inp_list = [(F_name, x_0, sig, noise_type, grad_eps, h, c1, c2, num_total_steps, seed, verbose, is_central) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_FD_GD(*inp)



    # Central FD
    if "CFD_GD" in OPT_TYPES:
        print("++++++ CFD ++++++")
        h = 1

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                is_central = True
                inp_list = [(F_name, x_0, sig, noise_type, grad_eps, h, c1, c2, num_total_steps, seed, verbose, is_central) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_FD_GD(*inp)




    # Our Method
    if "Our_GD" in OPT_TYPES:
        print("++++++ Our Method ++++++")
        coeff = 0
        max_h = 0.5

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                inp_list = [(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose, coeff, max_h, NUM_CPU) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    try:
                        run_our_GD(*inp)
                    except:
                        continue

                    # run_our_GD(*inp)

    

        
    # NEWUOA
    if "NEWUOA" in OPT_TYPES:
        print("++++++ NEWUOA ++++++")
        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                inp_list = [(F_name, x_0, sig, noise_type, grad_eps, c1, c2, num_total_steps, seed, verbose) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_NEWUOA(*inp)


