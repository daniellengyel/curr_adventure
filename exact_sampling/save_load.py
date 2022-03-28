from cgi import test
import json 
import os

def save_opt(opt_res, opt_type, test_problem_name, sig, noise_type, c1, c2, seed):
    save_name = "{}_{}_{}_{}_{}".format(noise_type, sig, c1, c2, seed)
    save_dir_path = "OptimizationResults/{}/{}".format(opt_type, test_problem_name)
    if not os.path.isdir(save_dir_path):
        os.makedirs(save_dir_path)
    with open(save_dir_path + "/" + save_name, "w") as f:
        json.dump(opt_res, f)

def load_opt(opt_type, test_problem_name, sig, noise_type, c1, c2, seed):
    save_name = "{}_{}_{}_{}_{}".format(noise_type, sig, c1, c2, seed)
    save_dir_path = "OptimizationResults/{}/{}".format(opt_type, test_problem_name)
    if not os.path.exists(save_dir_path + "/" + save_name):
        return None
    with open(save_dir_path + "/" + save_name, "r") as f:
        d = json.load(f)
    return d


