import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import scipy 

from jax.config import config
config.update("jax_enable_x64", True)



def loss_getter(dim, D, sig, coeff=0.1):
    def helper(X):
        S = X.reshape(dim, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ D @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        
        return 1/2 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S, ord="fro")**4
    return helper

def get_alpha(D, l1, l2):
    Dmax = jnp.max(D)
    Dmin = jnp.min(D)
    return -1/(Dmax - Dmin) * (Dmin + l2/(l1 - l2) * jnp.sum(D)) 

def get_V_star(dim, alpha):
    I = jnp.eye(dim)
    v1 = jnp.sqrt(I[0] + alpha * (I[-1] - I[0]))
    
    V_tmp = jnp.concatenate([v1.reshape(1, -1), I[1:]]).T
    V, _ = jnp.linalg.qr(V_tmp)
    return V

def get_lambda_star(dim, sig, coeff):
    
    if dim == 3:
        l1, l2 = jnp.cbrt(sig**2 / coeff), jnp.cbrt(sig**2 / (2*coeff))
    else:
        l1_denom = coeff*sig**2
        
        l1_factor = jnp.sqrt(coeff**2 * (dim + 1) * sig**4)
        
        
        cbrt_term = jnp.cbrt(coeff * sig**2 * (dim - 1 - jnp.sqrt(dim + 1)) / (2 * coeff**2 * dim * (dim - 3)) )
        
        l1 = l1_factor / l1_denom * cbrt_term
        l2 = cbrt_term
    lambda_star = jnp.concatenate([jnp.array([l1]), l2 * jnp.ones(dim - 1)])
    return lambda_star

def get_lambda_tilde(D_diag, sig, coeff, lmbda_star, eps_bound=1e-5):
    dim = len(D_diag)
    lmbdas_init = np.ones(dim)
    bounds = tuple((eps_bound, None) for _ in range(dim))

    a = D @ lmbda_star / len(lmbda_star)
    if a < 0:
        l_max_idx = jnp.argmax(D)
    else:
        l_max_idx = jnp.argmin(D)

    def lmbda_loss(lmbdas):
        a = D_diag @ lmbdas / len(lmbdas)
        b = jnp.sum(lmbdas)
        return 1/2 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas) + coeff*b**2

    lmbda_tilde = jnp.array(scipy.optimize.minimize(lmbda_loss, lmbdas_init, bounds=bounds)["x"])
    return lmbda_tilde

def get_multi_lambda_tilde(D_diag_multi, sig, coeff, lmbda_star_multi, eps_bound=1e-5):
    # bounds = tuple((eps_bound, None) for _ in range(dim))

    a_multi = [D_diag_multi[i] @ lmbda_star_multi[i] / len(lmbda_star_multi[i]) for i in range(len(D_diag_multi))]
    l_max_idx_multi = [jnp.argmax(D_diag_multi[i]) if a_multi[i] < 0 else jnp.argmin(D_diag_multi[i]) for i in range(len(D_diag_multi))]

    total_dim = sum([len(D_diag_multi[i]) for i in range(len(D_diag_multi))])

    def lmbda_loss(lmbdas):
        curr_loss = 0
        curr_lmbda_num = 0
        for i in range(len(D_diag_multi)):
            dim = len(D_diag_multi[i])
            a = D_diag_multi[i] @ lmbdas[curr_lmbda_num:curr_lmbda_num+dim] / dim
            b = jnp.sum(lmbdas[curr_lmbda_num:curr_lmbda_num+dim])
            curr_loss += 1/2 * a**2 * dim / lmbdas[curr_lmbda_num + l_max_idx_multi[i]] + sig**2 * dim/lmbdas[curr_lmbda_num + l_max_idx_multi[i]] + sig**2 * jnp.sum(1/lmbdas[curr_lmbda_num:curr_lmbda_num+dim]) + coeff*b**2
            curr_lmbda_num += dim
        return curr_loss

    lmbdas_init = np.ones(total_dim)
    bounds = tuple((eps_bound, None) for _ in range(total_dim))
    
    lmbda_tilde = jnp.array(scipy.optimize.minimize(lmbda_loss, lmbdas_init, bounds=bounds)["x"])
    lambda_tilde_multi = []
    curr_lmbda_num = 0
    for i in range(len(D_diag_multi)):
        dim = len(D_diag_multi[i])
        lambda_tilde_multi.append(lmbda_tilde[curr_lmbda_num:curr_lmbda_num+dim])
        curr_lmbda_num += dim

    return lambda_tilde_multi


def permute_rows(M, i, j):
    tmp_row = M[i].copy()
    M[i] = M[j].copy()
    M[j] = tmp_row
    return M


# This function may change if we decide to allow for lambda_star solutions. 
def generate_sing_vals_V(D_diag, sig, coeff):
    dim = len(D_diag) 

    lambda_star = get_lambda_star(dim, sig, coeff)
    lmbda = get_lambda_tilde(D_diag, sig, coeff, lambda_star, eps_bound=1e-8)
    sing_vals = jnp.diag(lmbda**0.5)
    V = jnp.eye(dim)

    return sing_vals, V


# This function may change if we decide to allow for lambda_star solutions. 
def generate_sing_vals_V_multi(D_diag_multi, sig, coeff):


    lambda_star_multi = [get_lambda_star(len(D_diag_multi[i]), sig, coeff) for i in range(len(D_diag_multi))]
    lmbda_multi = get_multi_lambda_tilde(D_diag_multi, sig, coeff, lambda_star_multi, eps_bound=1e-8)
    sing_vals = [lmbda_multi[i]**0.5 for i in range(len(D_diag_multi))]
    V_multi = [jnp.eye(len(D_diag_multi[i])) for i in range(len(D_diag_multi))]

    return sing_vals, V_multi





if __name__ == "__main__":
    D_diag_multi = [jnp.array([1, 200, -3, -4]), jnp.array([5, 6, 7, 8, 9, 10, 11, 12])]
    sig = 0.1
    coeff = 0.1
    print(generate_sing_vals_V_multi(D_diag_multi, sig, coeff)[0])
