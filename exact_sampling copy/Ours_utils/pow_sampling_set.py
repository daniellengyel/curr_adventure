
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import math

import time


from generate_sing_vals_V import generate_sing_vals_V

from jax.config import config
config.update("jax_enable_x64", True)


def get_S_pow_index_sets(D_diag,):
    """Returns a set of sets. Each set with the indecies of H to use."""
    dim = len(D_diag)
    binary_dim = (bin(dim)[2:])[::-1]
    res = {}
    for i in range(len(binary_dim)):
        if binary_dim[i] == "1":
            res[i] = []
            
    bin_used = sorted(res.keys())
    
    curr_n = len(bin_used) - 1
    upper_n = curr_n + 1
    lower_n = 0
    
    start = 0
    end = len(D_diag) - 1
    
    argsorted_D_diag = jnp.argsort(D_diag)
    while end < len(D_diag) and start <= end:
        while len(res[bin_used[curr_n]]) == 2**bin_used[curr_n]:
            curr_n -= 1
            curr_n = curr_n % upper_n
            
        curr_exp = bin_used[curr_n]
        if curr_exp == 0:
            res[curr_exp].append(int(argsorted_D_diag[start]))
            start += 1
        else:
            res[curr_exp].append(int(argsorted_D_diag[start]))
            res[curr_exp].append(int(argsorted_D_diag[end]))
            start += 1
            end -=1

        curr_n -= 1
        curr_n = curr_n % upper_n

    return res

def generate_all_pow_U(n):
    """Return matrix with dim 2^n x 2^n"""
    if n == 0:
        return [jnp.array([[1]])]
    pow_m = generate_all_pow_U(n - 1)
    sub_m = pow_m[-1]
    curr_m = jnp.concatenate([jnp.concatenate([sub_m, sub_m], axis=1), jnp.concatenate([sub_m, -sub_m], axis=1)])
    pow_m.append(curr_m / jnp.sqrt(2))
    return pow_m

def permute_rows(M, i, j):
    tmp_row = M[i].copy()
    M[i] = M[j].copy()
    M[j] = tmp_row
    return M

def create_approx_S(H, sig, max_h):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = np.linalg.eigh(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))

    D_diag = jnp.diag(D)

    S_pow_index_set = get_S_pow_index_sets(D_diag)
    all_pow_U = generate_all_pow_U(max(S_pow_index_set.keys()))
    S = np.zeros(shape=(dim, dim, ))
    for i in range(len(all_pow_U)):
        if i in S_pow_index_set:
            curr_index_set = jnp.array(S_pow_index_set[i])
            curr_sing_vals, _ = generate_sing_vals_V(D_diag[curr_index_set], sig, max_h)
            curr_U = permute_rows(np.array(all_pow_U[i]), 0, jnp.argmax(jnp.diag(curr_sing_vals)))
            curr_pow_S = np.array(curr_sing_vals @ curr_U)
            S[curr_index_set.reshape(-1, 1), curr_index_set] = curr_pow_S

    S = jnp.array(S)
    S = U_D @ S

    return S





