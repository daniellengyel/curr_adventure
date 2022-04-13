import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import scipy 

import matplotlib.pyplot as plt

from create_W import optimize_W, convert_to_U


from tqdm import tqdm
import time



from jax.config import config
config.update("jax_enable_x64", True)

def get_lambda_star(dim, sig, coeff):
    
    if dim == 3:
        return jnp.cbrt(2 * coeff * sig**2), jnp.cbrt(coeff * sig**2 / 4)
    
    l1_denom = coeff*sig**2
    
    l1_factor = jnp.sqrt(coeff**2 * (dim + 1) * sig**4)
    
    
    cbrt_term = jnp.cbrt(coeff * sig**2 * (dim - 1 - jnp.sqrt(dim + 1)) / (dim * (dim - 3)) )
    
    l1 = l1_factor / l1_denom * cbrt_term
    l2 = cbrt_term
    return l1, l2 

def get_lambda_tilde(D, sig, coeff, eps_bound=1e-5):
    dim = len(D)
    lmbdas_init = np.ones(dim)
    bounds = tuple((eps_bound, None) for _ in range(dim))
    
    def fun(lmbdas):
        a = D @ lmbdas / len(lmbdas)
        b = jnp.sum(lmbdas)
        return 1/2 * a**2 * dim / lmbdas[0] + sig**2 * dim/lmbdas[0] + sig**2 * jnp.sum(1/lmbdas) + lmbdas[0]**2 * dim#+ coeff*b**2

    return jnp.array(scipy.optimize.minimize(fun, lmbdas_init, bounds=bounds)["x"])

def loss_getter(dim, N, H, sig, coeff=0.1):
    def helper(X):

        S = X.reshape(N, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ H @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        
        return 1/2 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S)**4


    return helper

def lmbda_loss(lmbdas, dim, D, sig, coeff):
    a = D @ lmbdas / len(lmbdas)
    b = jnp.sum(lmbdas)
    return 1/2 * a**2 * dim / lmbdas[0] + sig**2 * dim/lmbdas[0] + sig**2 * jnp.sum(1/lmbdas) + coeff*b**2


# def get_alpha(D, l1, l2):
#     Dmax = jnp.max(D)
#     Dmin = jnp.min(D)
#     return 1/(Dmax - Dmin) * (Dmax + l2/(l1 - l2) * jnp.sum(D)) 


def create_S(H, sig, num_iter=10, x_init=None, coeff=0.1):
    dim = H.shape[0] 



    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = jnp.linalg.eig(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))
    # print(repr(jnp.diag(D)))
    # D = D + 0.01 * jnp.eye(dim)
    # print(D)


    diag_D = jnp.diag(D)
    # increasing_D
    increasing_sort = i = diag_D.argsort() 
    increasing_D = diag_D[increasing_sort]
    decreasing_D = increasing_D[::-1]

    start_time = time.time()

    lmbdas_increasing_tilde = get_lambda_tilde(increasing_D, sig, coeff, eps_bound=1e-4)
    lmbdas_decreasing_tilde = get_lambda_tilde(decreasing_D, sig, coeff, eps_bound=1e-4)

    loss_increasing = lmbda_loss(lmbdas_increasing_tilde, dim, increasing_D, sig, coeff)
    loss_decreasing = lmbda_loss(lmbdas_decreasing_tilde, dim, decreasing_D, sig, coeff)

    print("Time to get lmbdas and loss", time.time() - start_time)
    start_time = time.time()

    if loss_increasing < loss_decreasing:
        lmbdas_tilde = lmbdas_increasing_tilde[increasing_sort.argsort()]
    else:
        lmbdas_tilde = lmbdas_decreasing_tilde[increasing_sort[::-1].argsort()]

    sing_vals = lmbdas_tilde**0.5
    c = lmbdas_tilde * diag_D
    print(repr(lmbdas_tilde))
    min_D = diag_D.argmin()
    # print(repr(np.delete(c, min_D)))
    W, l_hist = optimize_W(np.delete(c, min_D), num_iter, x_init=x_init) 
    print(l_hist[-1])
    print("Time to get W", time.time() - start_time)
    # plt.plot(l_hist)
    # plt.show()
    U = convert_to_U(W, min_D)
    S = jnp.diag(sing_vals) @ U
    return U_D @ S, W



if __name__ == "__main__":
    from jax import grad

    dim = 3
    # D = np.linspace(1, 100, dim)
    # np.random.shuffle(D)
    # D = np.diag(D)
    D = jnp.diag(jnp.array([75.25,  -50.5, 25.75, 100, 200, 12, 89, 12]))
    sig = 0.1
    dim = len(D)
    coeff = 1

    l = loss_getter(dim, dim, D, sig, coeff=coeff)

    S, W = create_S(D, sig, num_iter=100, coeff=coeff)
    print(repr(S))
    print(l(S.T.flatten()))
    print(grad(l)(S.T.flatten()))




