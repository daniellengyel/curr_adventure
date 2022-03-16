import numpy as np
import matplotlib.pyplot as plt


import sys, os, time
import pickle
import yaml

import psutil 

import curr_adventure as curr_adv

# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
# config.update('jax_disable_jit', True)

# Job specific 
try:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
except:
    ARRAY_INDEX = -1

def experiment_run(config_inp, path):
    # setup saveing and save config
    file_stamp = str(time.time())
    exp_folder = os.path.join(path, file_stamp)
    os.mkdir(exp_folder)
    curr_adv.save_load.save_config(path, file_stamp, config_inp)

    # get opt and save
    a = time.time()
    results = curr_adv.main.optimize(config_inp, verbose=True)
    print(time.time() - a)
    # print(results)
    curr_adv.save_load.save_opt_path(path, file_stamp, results)


config = {}

# TODO Hyperparameter search over barrier noise and h 
# TODO Assess derivative for the ackley function
# TODO find problems with ill-conditioned Hessian

# particle init 
dim = 2
config["dim"] = dim
config["particle_init"] = "origin"



# function

config["Optimization_Problem"] = {"name": "Linear", 
                                    "num_barriers": dim*3,
                                    "obj_direction": "ones",
                                    "barrier_noise": 0.05}


# optimization
config["optimization_type"] = "IPM"
config["optimization_name"] = "BFGS"

config["optimization_meta"] = {"c1": 0.001, "c2": 0.7, "delta": 1, "N": dim*2, "h": 0.7, "jrandom_key": 1}
config["grad_estimate_type"] = "FD_Hess" #  "Exact" # "FD_uniform" # "FD_Hess" # 
# meta parameters (seed, how to save etc.)
config["seed"] = 0
config["return_full_path"] = True
config["num_path_steps"] = 15
config["num_total_steps"] = 300 

# --- Set up folder in which to store all results ---
folder_name = curr_adv.save_load.get_file_stamp(config["Optimization_Problem"]["name"])
cwd = os.environ["PATH_TO_ADV_FOLDER"]
folder_path = os.path.join(cwd, "experiments", "IPM", folder_name)
print(folder_path)
os.makedirs(folder_path)

# msg = "Use FSM on the symmetric function."
# with open(os.path.join(folder_path, "description.txt"), "w") as f:
#     f.write(msg)

analysis = experiment_run(config, folder_path) #tune.run(lambda config_inp:  experiment_run(config_inp, folder_path), config=config)
print(analysis)
