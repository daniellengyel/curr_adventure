import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generate_sing_vals_V import generate_sing_vals_V_multi, generate_sing_vals_V

from jax.config import config
config.update("jax_enable_x64", True)

import time

def get_S_pow_index_sets(dim, jrandom_key):
    """Returns a set of sets. Each set with the indecies of H to use."""
    curr_n = 0
    res = []
    indecies = jnp.array(range(dim))
    indecies = jrandom.permutation(jrandom_key, indecies, independent=True)
    curr_i = 0
    while dim > 0:
        if dim % 2 == 1:
            dim = (dim - 1)/2
            res.append(jnp.sort(indecies[curr_i:curr_i + 2**curr_n]))
            curr_i += 2**curr_n
            curr_n += 1
        else:
            curr_n += 1
            dim = dim/2
            res.append(None)
    return res

def generate_all_pow_U(n):
    """Return matrix with dim 2^n x 2^n"""
    if n == 1:
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

def create_approx_S(H, sig, coeff, jrandom_key):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = np.linalg.eigh(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))

    D_diag = jnp.diag(D)

    S_pow_index_set = get_S_pow_index_sets(dim, jrandom_key)
    all_pow_U = generate_all_pow_U(len(S_pow_index_set))
    S = np.zeros(shape=(dim, dim, ))

    all_sing_vals = []
    for i in range(len(all_pow_U)):
        if S_pow_index_set[i] is not None:
            curr_sing_vals, _ = generate_sing_vals_V(D_diag[S_pow_index_set[i]], sig, coeff)
            all_sing_vals.append(jnp.diag(curr_sing_vals))
            curr_U = permute_rows(np.array(all_pow_U[i]), 0, jnp.argmax(jnp.diag(curr_sing_vals)))
            curr_pow_S = np.array(curr_sing_vals @ curr_U)
            S[S_pow_index_set[i].reshape(-1, 1), S_pow_index_set[i]] = curr_pow_S
    print(all_sing_vals)
    S = jnp.array(S)
    S = U_D @ S

    return S


def eff_create_approx_S(H, sig, coeff, jrandom_key):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = np.linalg.eigh(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))

    D_diag = jnp.diag(D)

    S_pow_index_set = get_S_pow_index_sets(dim, jrandom_key)
    all_pow_U = generate_all_pow_U(len(S_pow_index_set))
    S = np.zeros(shape=(dim, dim, ))
    
    D_diag_multi = []
    for i in range(len(all_pow_U)):
        if S_pow_index_set[i] is not None:
            D_diag_multi.append(D_diag[S_pow_index_set[i]])
    
    print(D_diag_multi)
    start_time = time.time()

    sing_vals_multi, _ = generate_sing_vals_V_multi(D_diag_multi, sig, coeff)
    print(sing_vals_multi)
    print(time.time() - start_time)
    
    sing_vals_index = 0
    for i in range(len(all_pow_U)):
        if S_pow_index_set[i] is not None:
            curr_sing_vals = jnp.diag(sing_vals_multi[sing_vals_index])
            sing_vals_index += 1
            curr_U = permute_rows(np.array(all_pow_U[i]), 0, jnp.argmax(jnp.diag(curr_sing_vals)))
            curr_pow_S = np.array(curr_sing_vals @ curr_U)
            S[S_pow_index_set[i].reshape(-1, 1), S_pow_index_set[i]] = curr_pow_S
    S = jnp.array(S)
    S = U_D @ S

    return S

if __name__ == "__main__":
    from jax import grad

    dim = 3
    # D = np.linspace(1, 100, dim)
    # np.random.shuffle(D)
    # D = np.diag(D)
    D = jnp.diag(jnp.array([75.25,  -50.5, 25.75, 100, 200, 12, 89]))
    sig = 0.1
    dim = len(D)
    coeff = 1
    jrandom_key = jrandom.PRNGKey(0)

    print(eff_create_approx_S(D, sig, coeff, jrandom_key))
    print(create_approx_S(D, sig, coeff, jrandom_key))





