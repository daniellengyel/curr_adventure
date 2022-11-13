from audioop import add
from cgi import test
import pickle 
import os
HOME = os.getenv("HOME")
import time

import pandas as pd

def save_opt(opt_res, opt_res_X, opt_type, test_problem_name, dim, sig, noise_type, step_size, seed, problem_type, additional_info=None):
    test_problem_name = "{}_{}".format(test_problem_name, dim)

    save_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/{}/{}/{}".format(problem_type, opt_type, test_problem_name)

    if opt_type in ["GD", "NewtonsMethod"]:
        save_name = "{}".format(float(step_size))
        save_name = save_dir_path + "/" + save_name 

    else:
        save_name = "{}_{}_{}".format(float(step_size), noise_type, float(sig))
        
        if additional_info is not None:
            for k in additional_info:
                save_name += "_{}_{}".format(k, float(additional_info[k]))
            save_name = save_dir_path + "/" + save_name + "/seed_{}".format(seed)

    if not os.path.isdir(save_name):
        os.makedirs(save_name)

    print(save_name)

    with open(save_name + "/vals.pkl", "wb") as f:
        pickle.dump(opt_res, f)

    with open(save_name + "/x_data.pkl", "wb") as f:
        pickle.dump(opt_res_X, f)

def load_opt(opt_type, test_problem_name, sig, noise_type, step_size, seed, problem_type, additional_info={}):
    load_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/{}/{}/{}/".format(problem_type, opt_type, test_problem_name)
    
    if opt_type in ["GD", "NewtonsMethod"]:
        load_name = "{}".format(float(step_size))
        load_name = load_dir_path + load_name
    else:
        load_name = "{}_{}_{}".format(float(step_size), noise_type, float(sig))
        
        if additional_info is not None:
            for k in additional_info:
                load_name += "_{}_{}".format(k, float(additional_info)[k])
        
        load_name = load_dir_path + load_name + "/seed_{}".format(seed)
    
    if not os.path.exists(load_name):
        return None, None
    
    with open(load_name + "/vals.pkl", "rb") as f:
        d_res = pickle.load(f)
        
    with open(load_name + "/x_data.pkl", "rb") as f:
        d_x_data = pickle.load(f)
        
    return d_res, d_x_data

def convert_file_name(opt_type, file_name):
    file_name_split = file_name.split("_")
    
    res = {}
    
    if opt_type in ["GD", "NewtonsMethod"]:
        common_args = ["step_size"] 
    else:
        common_args = ["step_size", "noise_type", "sig"] 

    
    for i in range(len(common_args)):
        res[common_args[i]] = file_name_split[i]
        
    i = len(common_args)
    while i < len(file_name_split):
        if file_name_split[i] == "max":
            res["max_h"] = file_name_split[i + 2]
            i += 3
        else:
            res[file_name_split[i]] = file_name_split[i + 1]
            i += 2

    return res
    


def load_df():
    res_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/Quadratic/"
    
    res_dict = {}
    
    for opt_type in os.listdir(res_dir_path):
        if opt_type == ".DS_Store":
            continue
            
        for F_name in os.listdir(res_dir_path + opt_type):
            if (F_name, opt_type) not in res_dict:
                res_dict[(F_name, opt_type)] = []
        
            for file_name in os.listdir(res_dir_path + opt_type + "/" + F_name):
                
                if os.path.isdir(res_dir_path + opt_type + "/" + F_name + "/" + file_name):
                    curr_d = convert_file_name(opt_type, file_name)
                    res_dict[(F_name, opt_type)].append(curr_d)
                    
    pd_res_dict = {}
        
    for k, v in res_dict.items():
        pd_res_dict[k] = pd.DataFrame(v)
    
    return pd.concat(pd_res_dict).sort_index()
        
def load_all_opt(opt_type, test_problem_name, sig, noise_type, step_size, problem_type, additional_info={}):
    load_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/{}/{}/{}/".format(problem_type, opt_type, test_problem_name)
    
    if opt_type in ["GD", "NewtonsMethod"]:
        return load_opt(opt_type, test_problem_name, sig, noise_type, step_size, seed="None", additional_info={})
    
    load_name = "{}_{}_{}".format(float(step_size), noise_type, float(sig))
    
    if additional_info is not None:
        for k in additional_info:
            load_name += "_{}_{}".format(k, float(additional_info[k]))

    load_name = load_dir_path + load_name + "/all"
    
    print(load_name)

    if os.path.exists(load_name):
        with open(load_name + "/all_vals.pkl", "rb") as f:
            d_res = pickle.load(f)
        
        with open(load_name + "/all_x_data.pkl", "rb") as f:
            d_x_data = pickle.load(f)
    else:
        return None, None
    
    return d_res, d_x_data
