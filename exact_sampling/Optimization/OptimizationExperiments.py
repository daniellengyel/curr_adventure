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
from FD import FD
from NEWUO_test import NEWUOA_Wrapper

from multiprocessing import Pool
import multiprocessing

NUM_CPU = 1 # int(os.getenv("NUM_CPU"))
ARRAY_INDEX = os.getenv("PBS_ARRAY_INDEX")
if ARRAY_INDEX is None:
    ARRAY_INDEX = 1
else:
    ARRAY_INDEX = int(ARRAY_INDEX)
    
NUM_ARRAY = 50

def run_gd_approx_exp(opt_type, F_name, x_0, sig, noise_type, grad_eps, step_size, num_total_steps, seed, verbose, param_dict={}):
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)
    
    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)

    sig = F.sig
    noise_type = F.noise_type

    if opt_type == "OurMethod_GD":
        grad_getter = pow_SG(sig, max_h=param_dict["h"], NUM_CPU=1)
    elif opt_type == "CFD_GD":
        grad_getter = FD(sig, is_central=True, h=param_dict["h"]) 
    elif opt_type == "FD_GD":
        grad_getter = FD(sig, is_central=False, h=param_dict["h"]) 
    elif opt_type == "AdaptFD_GD":
        grad_getter = adapt_FD(sig, rl=1.5, ru=6) 

    optimizer = BFGS(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)    
    final_X, res, res_X = optimizer.run_opt()
    save_opt(res, res_X, opt_type, F_name, dim, sig, noise_type, step_size, seed, param_dict)
        
        
def run_NEWUOA(F_name, x_0, sig, noise_type, seed):
    jrandom_key = jrandom.PRNGKey(seed=seed)
    dim = len(x_0)

    F_name, x_0, F = PyCutestGetter(func_name=F_name, func_dim=len(x_0), sig=sig, noise_type=noise_type)
    curr_F_inst = NEWUOA_Wrapper(F, jrandom_key)
    curr_F = lambda X: float(curr_F_inst.f(X))

    newuoa_full_res = newuoa(curr_F, x_0) 
    newuoa_res = newuoa_full_res["fhist"]
    step_size = 0
    save_opt(newuoa_res, None, "NEWUOA", F_name, dim, sig, noise_type, step_size, seed)


if __name__ == "__main__":

    OPT_TYPES = os.getenv("OPT_TYPES").split(" ")

    sig = float(os.getenv("SIG"))
    step_size = float(os.getenv("STEP_SIZE"))
    noise_type="uniform"

    num_total_steps = 50
    grad_eps = 1e-5

    verbose = False

    test_problem_iter = range(85)
    dim_iter = range(0, 10)

    num_trials = 50
    lower_seed, upper_seed = int(num_trials / NUM_ARRAY) * (ARRAY_INDEX - 1), min(int(num_trials / NUM_ARRAY) * (ARRAY_INDEX ), num_trials)

    h = float(os.getenv("h"))

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

                optimizer = NewtonMethod(x_0, F, step_size, num_total_steps, newton_sig, None, grad_eps, verbose=verbose)
                final_X, exact_res, exact_res_X = optimizer.run_opt()
                save_opt(exact_res, exact_res_X, "NewtonsMethod", F_name, len(x_0), newton_sig, noise_type, step_size, None)

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

                optimizer = GradientDescent(x_0, F, step_size, num_total_steps, GD_sig, None, grad_eps, verbose=verbose)
                final_X, exact_res, exact_res_X = optimizer.run_opt()
                save_opt(exact_res, exact_res_X, "GradientDescent", F_name, len(x_0), GD_sig, noise_type, step_size, None)


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

                inp_list = [("AdaptFD_GD", F_name, x_0, sig, noise_type, grad_eps, step_size, num_total_steps, seed, verbose) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_gd_approx_exp(*inp)


    # Forward FD
    if "FD_GD" in OPT_TYPES:
        print("++++++ FD ++++++")

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                is_central = False
                inp_list = [("FD_GD", F_name, x_0, sig, noise_type, grad_eps, step_size, num_total_steps, seed, verbose, {"h": h}) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_gd_approx_exp(*inp)


    # Central FD
    if "CFD_GD" in OPT_TYPES:
        print("++++++ CFD ++++++")

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                is_central = True
                inp_list = [("CFD_GD", F_name, x_0, sig, noise_type, grad_eps, step_size, num_total_steps, seed, verbose, {"h": h}) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_gd_approx_exp(*inp)


    # Our Method
    if "Our_GD" in OPT_TYPES:
        print("++++++ Our Method ++++++")

        for f_i in tqdm(test_problem_iter):
            for d_i in tqdm(dim_iter):
                F_name, x_0, F = PyCutestGetter(func_i=f_i, dim_i=d_i, sig=sig, noise_type=noise_type)
                
                if F is None:
                    break
                
                print("F Name", F_name)
                print("F Dim", len(x_0))

                inp_list = [("OurMethod_GD", F_name, x_0, sig, noise_type, grad_eps, step_size, num_total_steps, seed, verbose, {"h": h}) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    try:
                        run_gd_approx_exp(*inp)
                    except:
                        continue

                    # run_gd_approx_exp(*inp)

    

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

                inp_list = [(F_name, x_0, sig, noise_type, seed) for seed in range(lower_seed, upper_seed)]
                for inp in inp_list:
                    run_NEWUOA(*inp)


