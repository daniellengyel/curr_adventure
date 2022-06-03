import jax.numpy as jnp
import jax.random as jrandom
from Optimization import ExactH_GD, InterpH_GD

from jax.config import config
config.update("jax_enable_x64", True)

from save_load import save_opt
from tqdm import tqdm

import math
import sys 
import os 
HOME = os.getenv("HOME")
sys.path.append(HOME + "/curr_adventure/exact_sampling/")

from Functions import get_F

from AdaptiveFD import adapt_FD
from FD import FD
from pow_sampling_set import pow_SG
from ExactGrad import ExactGrad

ARRAY_INDEX = os.getenv("PBS_ARRAY_INDEX")
if ARRAY_INDEX is None:
    ARRAY_INDEX = 1
else:
    ARRAY_INDEX = int(ARRAY_INDEX)

ARRAY_INDEX -= 1
    
NUM_ARRAY = 15

def run_exp(F_type, F_name, sig, noise_type, opt_type, grad_eps, step_size, num_total_steps, seed, verbose=False, param_dict={}):
    
    F, x_0 = get_F(F_type, F_name, sig, noise_type)
    dim = len(x_0)

    jrandom_key = jrandom.PRNGKey(seed=seed)

    opt_specs = opt_type.split("_")

    if len(opt_specs) == 2:
        H_get_type = opt_specs[0]
        general_opt_type = opt_specs[1]
    else:
        H_get_type = None
        general_opt_type = opt_specs[0]

    if general_opt_type == "Ours":
        grad_getter = pow_SG(sig, max_h=param_dict["h"], NUM_CPU=1)
    elif general_opt_type == "CFD":
        grad_getter = FD(sig, is_central=True, h=param_dict["h"], use_H=False) 
    elif general_opt_type == "FD":
        if H_get_type is not None: 
            grad_getter = FD(sig, is_central=False, h=param_dict["h"], use_H=True) 
        else: 
            grad_getter = FD(sig, is_central=False, h=param_dict["h"], use_H=True) 
    elif general_opt_type == "AdaptFD":
        grad_getter = adapt_FD(sig, rl=1.5, ru=6) 
    elif general_opt_type == "GD":
        grad_getter = ExactGrad()

    if (H_get_type is not None) and (H_get_type != "Exact"):
        optimizer = InterpH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose, smoothing=param_dict["smoothing"])    
    else:
        optimizer = ExactH_GD(x_0, F, step_size, num_total_steps, sig, jrandom_key, grad_getter, grad_eps, verbose=verbose)  

    _, res, res_X = optimizer.run_opt()
    save_opt(res, res_X, opt_type, F_name, dim, sig, noise_type, step_size, seed, "Quadratic", param_dict)

        


if __name__ == "__main__":



    num_total_steps = 500
    grad_eps = 1e-5

    verbose = False

    noise_type="uniform"
    num_trials = 50
    SIGS = [1]
    STEP_SIZES = [1e-1, 1e-2]
    HS = [5e-3, 0.1, 0.5, 1.0, 2.0]
    SMOOTHINGS = [0, 1, 5, 20]
    SEEDS = list(range(num_trials))
    OPT_TYPES = ["GD", "FD", "CFD", "AdaptFD", "Interp_Ours", "Exact_FD", "Interp_FD", "Exact_Ours"]
    F_TYPE = "Logistic"
    F_NAMES = ["Heart"] #["{}_{}_{}_{}_{}".format(10, "log", -2, 4, 0)] # dim, interp_type, lw, ub, seed

    # GD = len(STEP_SIZES)
    # ADAPT = GD * len(SEEDS) * len(SIGS) 
    # FD = 2 * ADAPT * len(HS)
    # EXACT = 2 * ADAPT * len(HS)
    # INTERP = 2 * ADAPT * len(HS) * len(SMOOTHINGS)
    # print(GD + ADAPT + FD + EXACT + INTERP)

    inp_list = []

    for opt_type in OPT_TYPES:
        for step_size in STEP_SIZES:
            for F_name in F_NAMES:
                if opt_type == "GD":
                    sig = 0
                    seed = 0
                    inp_list.append((F_TYPE, F_name, sig, noise_type, opt_type, grad_eps, step_size, num_total_steps, seed, verbose, {}))
                else:
                    for sig in SIGS:
                        for seed in SEEDS:
                            if opt_type == "AdaptFD":
                                inp_list.append((F_TYPE, F_name, sig, noise_type, opt_type, grad_eps, step_size, num_total_steps, seed, verbose, {}))
                            else:
                                for h in HS:
                                    if "Interp" == opt_type.split("_")[0]:
                                        for smoothing in SMOOTHINGS:
                                            inp_list.append((F_TYPE, F_name, sig, noise_type, opt_type, grad_eps, step_size, num_total_steps, seed, verbose, {"h": h, "smoothing": smoothing}))
                                    else:
                                        inp_list.append((F_TYPE, F_name, sig, noise_type, opt_type, grad_eps, step_size, num_total_steps, seed, verbose, {"h": h}))

    num_inps = len(inp_list)                                

    ub = min(math.ceil(num_inps / NUM_ARRAY) * (ARRAY_INDEX + 1), len(inp_list))
    lb = math.ceil(num_inps / NUM_ARRAY) * ARRAY_INDEX

    for i in tqdm(range(lb, ub)):
        run_exp(*inp_list[i])

