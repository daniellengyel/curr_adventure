

from tqdm import tqdm
import pickle

import os, shutil
HOME = os.getenv("HOME")



def convert_file_name(opt_type, file_name):
    file_name_split = file_name.split("_")
    
    res = {}
    
    
    common_args = ["noise_type", "sig", "c1", "c2"]


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


res_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/"

for opt_type in ["NEWUOA", "OurMethod_GD", "CentralFD_GD"]:

    for F_name in tqdm(os.listdir(res_dir_path + opt_type)):

        for exp_name in os.listdir(res_dir_path + opt_type + "/" + F_name):
            if not os.path.isdir(res_dir_path + opt_type + "/" + F_name + "/" + exp_name + "/all"):
                os.makedirs(res_dir_path + opt_type + "/" + F_name + "/" + exp_name + "/all")
            all_path = res_dir_path + opt_type + "/" + F_name + "/" + exp_name + "/all"

            for seed in os.listdir(res_dir_path + opt_type + "/" + F_name + "/" + exp_name):
                if seed == "all":
                    continue
                    
                seed_path = res_dir_path + opt_type + "/" + F_name + "/" + exp_name + "/" + seed
                            
                if ".pkl" in seed:
                    os.remove(seed_path)
                    continue
                            

                try:
                    with open(seed_path + "/vals.pkl", "rb") as f:
                        opt_res = pickle.load(f)

                    with open(seed_path + "/x_data.pkl", "rb") as f:
                        opt_res_X = pickle.load(f)
                except:
                    continue
                
                seed_num = seed.split("_")[1]
                            
                # save vals
                if os.path.exists(all_path + "/all_vals.pkl"):
                    with open(all_path + "/all_vals.pkl", "rb") as f:
                        d_res = pickle.load(f)
                else:
                    d_res = {}
                    
                if seed in d_res:
                    del d_res[seed]
                    
                if seed_num not in d_res:
                    
                    d_res[seed_num] = opt_res
                    with open(all_path + "/all_vals.pkl", "wb") as f:
                        pickle.dump(d_res, f)

                # save x data
                if os.path.exists(all_path + "/all_x_data.pkl"):
                    with open(all_path + "/all_x_data.pkl", "rb") as f:
                        d_res_X = pickle.load(f)
                else:
                    d_res_X = {}
                    
                if seed in d_res_X:
                    del d_res_X[seed]
                    
                if seed_num not in d_res_X:
                    
                    d_res_X[seed_num] = opt_res_X

                    with open(all_path + "/all_x_data.pkl", "wb") as f:
                        pickle.dump(d_res_X, f)

                
                