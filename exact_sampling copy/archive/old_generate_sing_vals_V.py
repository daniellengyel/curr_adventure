import jax.numpy as jnp
from jax import grad, jacfwd
import jax.random as jrandom
import numpy as np

import scipy 

from jax.config import config
config.update("jax_enable_x64", True)

import time

def loss_getter(dim, D, sig, coeff=0.1):
    def helper(X):
        S = X.reshape(dim, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ D @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        
        return 1/4 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S, ord="fro")**4
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

def lmbda_loss(lmbdas, D_diag, sig, coeff, lmbda_star):
    dim = len(D_diag)

    a = D_diag @ lmbda_star / len(lmbda_star)
    alpha = get_alpha(D_diag, lmbda_star[0], lmbda_star[1])
    if (alpha < 1) and (alpha > 0):
        l_max_idx = jnp.argmin(jnp.abs(D_diag))
    elif a < 0:
        l_max_idx = jnp.argmax(D_diag)
    else:
        l_max_idx = jnp.argmin(D_diag)

    a = D_diag @ lmbdas / len(lmbdas)
    b = jnp.sum(lmbdas)
    return 1/4 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas) + coeff*b**2

def lambda_coeff_zero(D_diag, l_max_idx, sig):
    dim = len(D_diag)
    D_diag = jnp.abs(D_diag)
    D_sum = jnp.sum(jnp.sqrt(D_diag)) - jnp.sqrt(D_diag[l_max_idx])
    D_l_max_idx = D_diag[l_max_idx]
    numerator = 2 * (1 + dim) * D_l_max_idx * sig**2 + D_sum**2 * sig**2 + jnp.sqrt(D_sum**2 * (8 * (1 + dim) * D_l_max_idx + D_sum**2) * sig**4)
    
    a = jnp.sqrt(2 * numerator/(dim * D_l_max_idx))
    l = (2 * (1 + dim) * D_l_max_idx * sig**2 + numerator)/(D_l_max_idx**2 * a)

    lmbdas = [(sig * jnp.sqrt(2*l/(D_diag[i] * a))) for i in range(len(D_diag))]
    lmbdas[jnp.argmin(jnp.abs(D_diag))] = l
    
    return jnp.array(lmbdas), [a, 0, l]

def lambda_coeff_zero_set_max_lambda(D_diag, l_max, l_max_idx, sig):
    dim = len(D_diag)
    D_diag = jnp.abs(D_diag)
    D_sum = jnp.sum(jnp.sqrt(D_diag)) - jnp.sqrt(D_diag[l_max_idx])
    D_l_max_idx = D_diag[l_max_idx]

    D_l_max_idx * l_max + dim * ((-D_l_max_idx**3 * l_max**3))


    lmbdas = [(sig * jnp.sqrt(2*l/(D_diag[i] * a))) for i in range(len(D_diag))]
    lmbdas[jnp.argmin(jnp.abs(D_diag))] = l
    
    return jnp.array(lmbdas), [a, 0, l_max]
    
def get_lambda_tilde(D_diag, sig, coeff, lmbda_star, eps_bound=1e-5):
    dim = len(D_diag)
    lmbdas_init = np.ones(dim)
    bounds = tuple((eps_bound, None) for _ in range(dim))

    a = D_diag @ lmbda_star / len(lmbda_star)
    alpha = get_alpha(D_diag, lmbda_star[0], lmbda_star[1])
    if (alpha < 1) and (alpha > 0):
        l_max_idx = jnp.argmin(jnp.abs(D_diag))
    elif a < 0:
        l_max_idx = jnp.argmax(D_diag)
    else:
        l_max_idx = jnp.argmin(D_diag)

    def lmbda_loss(lmbdas):
        a = D_diag @ lmbdas / len(lmbdas)
        b = jnp.sum(lmbdas)
        return 1/4 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas) + coeff * b**2

    g_l = grad(lmbda_loss)
    h_l = jacfwd(g_l)
    lmbda_tilde = jnp.array(scipy.optimize.minimize(lmbda_loss, lmbdas_init, method="Newton-CG", jac=g_l, hess=h_l)["x"])
    return lmbda_tilde



def get_lambda_tilde_least_squares(D_diag, sig, coeff, max_h, lmbda_star, prev_sol=None):
    
    dim = len(D_diag)
    a_star = D_diag @ lmbda_star / len(lmbda_star)
    alpha = get_alpha(D_diag, lmbda_star[0], lmbda_star[1])
    if (alpha < 1) and (alpha > 0):
        l_max_idx = jnp.argmin(jnp.abs(D_diag))
    elif a_star < 0:
        l_max_idx = jnp.argmax(D_diag)
    else:
        l_max_idx = jnp.argmin(D_diag)

    def helper(x):
        a, b, l = x[0], jnp.abs(x[1]), jnp.abs(x[2])
        
        D_seq = 1/jnp.sqrt(D_diag * a / (2*l) + b)
        # Go through this and double check. Also, for dimensions lower than lets say 32 it's probably better to use the minimization method. (Let's also just compare with mathematica.)
        f1 = 1/dim * (sig * (jnp.sum((D_diag * D_seq)[:l_max_idx]) + jnp.sum((D_diag * D_seq)[l_max_idx+1:])) + D_diag[l_max_idx] * l) - a
        f2 = 2 * coeff * (sig * (jnp.sum(D_seq[:l_max_idx]) + jnp.sum(D_seq[l_max_idx+1:])) + l) - b
        f3 = 1/2. * a * D_diag[l_max_idx] * l - 1/4. * a**2 * dim - sig**2 * (dim + 1) + b * l**2

        return jnp.array([f1, f2, f3])

    # if (prev_sol is not None) and (prev_sol != []):
    #     x_init = prev_sol
    # else:
    if coeff == 0:
        lmbda_tilde, _ = lambda_coeff_zero(D_diag, l_max_idx, sig)
        lmbda_tilde = jnp.clip(lmbda_tilde, a_max=max_h)
        # print(lmbda_tilde)
        return lmbda_tilde, _

    x_init = [0, 1, 1/(1 + D_diag[l_max_idx])]

    res = scipy.optimize.least_squares(helper, x_init, ) 
    # print(res)

    a, b, l = res["x"]
    b = jnp.abs(b)
    l = jnp.abs(l)
    
    lmbda_tilde = jnp.array([float(sig*jnp.sqrt(1/(a * D_diag[i] / (2*l) + b))) for i in range(l_max_idx)] + [l] + [float(sig*jnp.sqrt(1/(a * D_diag[i] / (2*l) + b))) for i in range(l_max_idx+1, dim)])
    

    return lmbda_tilde, res["x"]


def permute_rows(M, i, j):
    tmp_row = M[i].copy()
    M[i] = M[j].copy()
    M[j] = tmp_row
    return M


def generate_sing_vals_V(D_diag, sig, coeff, max_h, prev_sols=None):
    dim = len(D_diag)
    lambda_star = get_lambda_star(dim, sig, coeff)
    lmbda, curr_sols = get_lambda_tilde_least_squares(D_diag, sig, coeff, max_h, lambda_star, prev_sols)
    sing_vals = jnp.diag(lmbda**0.5)
    V = jnp.eye(dim)

    if prev_sols is None:
        return sing_vals, V
    else:
        return sing_vals, V, curr_sols



if __name__ == "__main__":
    pass