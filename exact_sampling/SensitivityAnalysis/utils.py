import jax.random as jrandom
import jax.numpy as jnp

from tqdm import tqdm 

import sys
import os 

HOME = os.environ["PATH_SIMPLEX_FOLDER"]
sys.path.append(HOME)

from FD import FD
from Ours import Ours
from RBF import RBF



def mse_sensitivity(F_true, F_tilde, sig, pts, rbf_f, h_space, jrandom_key, num_runs, true_h=1e-10):
#     h_space = jnp.logspace(-11, -9, 6)
    res_fd = [[[] for j in range(len(h_space))] for i in range(len(pts))]
    res_ours = [[[] for j in range(len(h_space))] for i in range(len(pts))]
    res_cfd = [[[] for j in range(len(h_space))] for i in range(len(pts))]
    res_rbf = []

    grad_getter_cfd_true = FD(sig=0, is_central=True, h=true_h)
    
    for k in range(num_runs):
        jrandom_key, subkey = jrandom.split(jrandom_key)
        for i in tqdm(range(len(pts))):
            p = pts[i]

            H = rbf_f.f2(p)
            true_grad = grad_getter_cfd_true.grad(F_true, p, jrandom_key=subkey, H=None)[0]
            if k == 0:
                res_rbf.append(jnp.linalg.norm(rbf_f.f1(p) - true_grad))
                # print(jnp.linalg.cond(H))
                # print(jnp.linalg.eigh(H)[0])

            for h_i, h in enumerate(h_space):

                grad_getter_fd = FD(sig=sig, is_central=False, h=h)
                grad_getter_ours = Ours(sig=sig, max_h=h)
                grad_getter_cfd = FD(sig=sig, is_central=True, h=h)

                res_fd[i][h_i].append(jnp.linalg.norm(grad_getter_fd.grad(F_tilde, p, subkey, H=H)[0] - true_grad))
                res_ours[i][h_i].append(jnp.linalg.norm(grad_getter_ours.grad(F_tilde, p, subkey, H=H)[0] - true_grad))
                res_cfd[i][h_i].append(jnp.linalg.norm(grad_getter_cfd.grad(F_tilde, p, subkey, H=H)[0] - true_grad))


    return jnp.array(res_fd), jnp.array(res_cfd), jnp.array(res_ours), jnp.array(res_rbf)
        

# Get all RBF
def get_rbfs(Fs, prmts_0, N_pts, prct_bound, jrandom_key, fixed_bound=None, smoothing=0):
    res = []
    
    
    jrandom_key, subkey = jrandom.split(jrandom_key)

    if fixed_bound is not None:
        prmts_fine = prmts_0.reshape(1, len(prmts_0)) + fixed_bound*(jrandom.uniform(subkey, shape=(N_pts, len(prmts_0))) - 0.5)*2
    else:
        pts_prct = prct_bound*(jrandom.uniform(subkey, shape=(N_pts, len(prmts_0))) - 0.5)*2
        prmts_fine = prmts_0.reshape(1, len(prmts_0))*(1 + pts_prct)

    for F in Fs:
        out = []
        jrandom_key, subkey = jrandom.split(jrandom_key)
        for p in tqdm(prmts_fine):
            out.append(F.f(p, subkey))

        rbf_f = RBF(prmts_fine.T, out, smoothing=smoothing)
        res.append(rbf_f)
        
    return res