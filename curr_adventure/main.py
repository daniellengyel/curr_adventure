import numpy as np

from .utils import *
from .Functions import *
from .IPMOptimizers import get_optimizer as get_ipm_opt

import time

"""Particle output will have the shape [[[]-particles]-timesteps] or paths[time_step][particle_idx]. Vals output has shape [[(obj, barrier, time)]-timestep]"""




def optimize(config, verbose=False):

    # init num_particles
    all_vals = []
    p_start = get_particles(config)
    p_curr = np.array([np.array(p) for p in p_start])

    num_steps = config["num_path_steps"]

    # get optimization method
    opt = get_ipm_opt(config)

    if config["seed"] is not None:
        np.random.seed(config["seed"])

    for t in range(num_steps):
        print(t)

        p_curr, obj_barrier_vals = opt.update(p_curr, t, record_vals=True)
        p_curr = p_curr.copy()
        all_vals.append(obj_barrier_vals)
        
    return all_vals
