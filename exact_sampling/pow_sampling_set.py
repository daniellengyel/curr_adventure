import multiprocessing
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import math

import time
from multiprocessing import Pool
import multiprocessing


from generate_sing_vals_V import generate_sing_vals_V
from simplex_gradient import simplex_gradient

from jax.config import config
config.update("jax_enable_x64", True)

# Only use for trying different choices for the pow_index_sets. 
def get_S_pow_index_sets_rand(D_diag, jrandom_key=None):
    """Returns a set of sets. Each set with the indecies of H to use."""
    jrandom_key = jrandom.PRNGKey(np.random.randint(0, 100))
    dim = len(D_diag)
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

def get_S_pow_index_sets(D_diag,):
    """Returns a set of sets. Each set with the indecies of H to use."""
    dim = len(D_diag)
    curr_n = int(math.log2(dim))
    res = []
    curr_start = 0
    curr_end = dim
    while curr_n >= 0:

        if dim >= 2**curr_n:
            dim = dim - 2**curr_n
            next_start = int(curr_start + 2**(curr_n-1))
            next_end = int(curr_end - 2**(curr_n-1)) # will round down when curr_n == 0 while the one above will not round up. so we are good. 
            res.append(jnp.array(list(range(curr_start, next_start)) + list(range(next_end, curr_end))))
            curr_start = next_start
            curr_end = next_end
            curr_n -= 1
        else:
            curr_n -= 1
            res.append(None)
    return res[::-1]


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

def create_approx_S(H, sig, coeff, prev_sols=None):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = np.linalg.eigh(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))

    D_diag = jnp.diag(D)

    S_pow_index_set = get_S_pow_index_sets(D_diag)
    all_pow_U = generate_all_pow_U(len(S_pow_index_set))
    S = np.zeros(shape=(dim, dim, ))

    if prev_sols == []:
        prev_sols = [[]] * len(all_pow_U)

    all_sols = []
    for i in range(len(all_pow_U)):
        if S_pow_index_set[i] is not None:
            if prev_sols is not None:
                curr_sing_vals, _, curr_sols = generate_sing_vals_V(D_diag[S_pow_index_set[i]], sig, coeff, prev_sols[i])
            else:
                curr_sing_vals, _ = generate_sing_vals_V(D_diag[S_pow_index_set[i]], sig, coeff)
                curr_sols = None
            all_sols.append(curr_sols)
            curr_U = permute_rows(np.array(all_pow_U[i]), 0, jnp.argmax(jnp.diag(curr_sing_vals)))
            curr_pow_S = np.array(curr_sing_vals @ curr_U)
            S[S_pow_index_set[i].reshape(-1, 1), S_pow_index_set[i]] = curr_pow_S
        else:
            all_sols.append(None)

    S = jnp.array(S)
    S = U_D @ S

    if prev_sols is None:
        return S
    else: 
        return S, all_sols


def helper_create_approx_S_multi(D_diag, sig, coeff, S, curr_S_pow_index_set, curr_all_pow_U):
    curr_sing_vals, _ = generate_sing_vals_V(D_diag[curr_S_pow_index_set], sig, coeff)
    curr_U = permute_rows(np.array(curr_all_pow_U), 0, jnp.argmax(jnp.diag(curr_sing_vals)))
    curr_pow_S = np.array(curr_sing_vals @ curr_U)
    S[curr_S_pow_index_set.reshape(-1, 1), curr_S_pow_index_set] = curr_pow_S
    return S


def create_approx_S_multi(H, sig, coeff, pool):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = np.linalg.eigh(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))

    D_diag = jnp.diag(D)

    S_pow_index_set = get_S_pow_index_sets(D_diag)
    all_pow_U = generate_all_pow_U(len(S_pow_index_set))

    pool_inp = []
    for i in range(len(all_pow_U)):
        if S_pow_index_set[i] is not None:
            pool_inp.append((D_diag, sig, coeff,  np.zeros(shape=(dim, dim, )), S_pow_index_set[i], all_pow_U[i]))

    if len(pool_inp) > 2 and pool is not None:
        res = pool.starmap(helper_create_approx_S_multi, pool_inp)
    else:
        res = [helper_create_approx_S_multi(*inp) for inp in pool_inp]

    S = np.zeros(shape=(dim, dim, ))
    for sub_S in res:
        S += sub_S
    S = jnp.array(S)
    S = U_D @ S

    return S

class pow_SG:
    def __init__(self, sig, coeff=1, max_h=2):
        self.sig = sig
        self.coeff = coeff
        self.max_h = max_h
        self.pool = Pool(processes=int(multiprocessing.cpu_count()/2.))

    def grad(self, F, X, jrandom_key, H):

        x_0 = X
        if len(x_0.shape) != 1:
            x_0 = x_0.reshape(-1)

        S = create_approx_S_multi(H, self.sig, self.coeff, self.max_h, self.pool) # optimize_uncentered_S(H, self.sig, self.coeff, max_steps=50) #
        return simplex_gradient(F, x_0, S, jrandom_key)


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


    print(create_approx_S(D, sig, coeff, jrandom_key))





