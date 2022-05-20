import os, shutil, pickle
HOME = os.getenv("HOME")

from tqdm import tqdm 

def save_seeds_to_all(exp_path, seeds):
    d_vals = {}
    d_x_data = {}
    try:
        if os.path.exists(exp_path + '/all'):
            with open(exp_path + '/all/all_vals.pkl', "rb") as f:
                d_vals = pickle.load(f)
            with open(exp_path + '/all/all_x_data.pkl', "rb") as f:
                d_x_data = pickle.load(f)
    except:
        d_vals = {}
        d_x_data = {}

    for seed in seeds:
        if seed not in d_vals:
            try:
                with open(exp_path + '/seed_{}'.format(seed) + "/vals.pkl", "rb") as f:
                    d_vals[seed] = pickle.load(f)
                with open(exp_path + '/seed_{}'.format(seed) + "/x_data.pkl", "rb") as f:
                    d_x_data[seed] = pickle.load(f)
            except:
                continue

    if not os.path.exists(exp_path + '/all/'):
        os.makedirs(exp_path + '/all/')

    with open(exp_path + '/all/all_vals.pkl', "wb") as f:
        d_vals = pickle.dump(d_vals, f)
    with open(exp_path + '/all/all_x_data.pkl', "wb") as f:
        d_x_data = pickle.dump(d_x_data, f)



res_dir_path = HOME + "/curr_adventure/exact_sampling/OptimizationResults/"

for opt_type in ["CFD_GD"]: # os.listdir(res_dir_path):
    if opt_type == ".DS_Store":
        continue
        
    print("Opt Type:", opt_type)

    for F_name in tqdm(os.listdir(res_dir_path + opt_type)):
    
        for exp_name in os.listdir(res_dir_path + opt_type + "/" + F_name):
            
            # get all seeds 
            curr_seeds = []
            for seed_file in os.listdir(res_dir_path + opt_type + "/" + F_name + "/" + exp_name):
                split_seed_file = seed_file.split("_")
                if len(split_seed_file) > 0 and split_seed_file[0] == "seed":
                    curr_seeds.append(split_seed_file[1])


            save_seeds_to_all(res_dir_path + opt_type + "/" + F_name + "/" + exp_name, curr_seeds)
                