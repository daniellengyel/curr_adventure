from audioop import add
from cgi import test
import pickle 
import os
HOME = os.getenv("HOME")
import time

def save_opt(opt_res, opt_res_X, opt_type, test_problem_name, dim, sig, noise_type, c1, c2, seed, additional_info=None):
    test_problem_name = "{}_{}".format(test_problem_name, dim)
    save_name = "{}_{}_{}_{}".format(noise_type, sig, c1, c2)
    if additional_info is not None:
        for k in additional_info:
            save_name += "_{}_{}".format(k, additional_info[k])
    save_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/{}/{}".format(opt_type, test_problem_name)
    if not os.path.isdir(save_dir_path):
        os.makedirs(save_dir_path)

    if not os.path.isdir(save_dir_path + "/" + save_name):
        os.makedirs(save_dir_path + "/" + save_name)

    seed_path = save_dir_path + "/" + save_name + "/seed_{}".format(seed)
    if not os.path.isdir(seed_path):
        os.makedirs(seed_path)

    with open(seed_path + "/vals.pkl", "wb") as f:
        pickle.dump(opt_res, f)

    with open(seed_path + "/x_data.pkl", "wb") as f:
        pickle.dump(opt_res_X, f)

def load_opt(opt_type, test_problem_name, sig, noise_type, c1, c2, seed, additional_info={}):
    save_name = "{}_{}_{}_{}".format(noise_type, sig, c1, c2)
    if additional_info is not None:
        for k in additional_info:
            save_name += "_{}_{}".format(k, additional_info[k])
    save_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/{}/{}/seed_{}".format(opt_type, test_problem_name, seed)
    if not os.path.exists(save_dir_path + "/" + save_name):
        return None
    with open(save_dir_path + "/" + save_name + ".pkl", "rb") as f:
        d = pickle.load(f)
    return d



