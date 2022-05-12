from venv import create
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax import grad

from tqdm import tqdm
import time

from jax.config import config
config.update("jax_enable_x64", True)

import sys 
sys.path.append("../")

from generate_sing_vals_V import generate_sing_vals_V
from U_opt_sampling_set import generate_regular_simplex, orthog_linesearch, convert_to_U

def optimize_W(A, P, sing_vals, V, U_D, num_iter, jrandom_key, loss, x_init=None):
    """Constraint is U.c = \bar{c} 1 and U.U^T = I"""
    
    dim = len(A)
    
    R = generate_regular_simplex(dim - 1)
    
    # init X
    if x_init is None:
        X = jnp.eye(dim - 1)
        # if jrandom_key is not None:
        #     jrandom_key, subkey = jrandom.split(jrandom_key)
        #     X = jrandom.normal(subkey, shape=(dim-1, dim-1))
        #     X = jnp.real(jnp.linalg.eig(X + X.T)[1])
    else:
        X = x_init
    I = jnp.eye(dim - 1)
    
    
    def l(U):
        U_matrix = jnp.sqrt((dim - 1)/dim) * U.reshape(dim - 1, dim - 1) @ R
        val_M = U_matrix.T @ A[1:, 1:] @ U_matrix + 1/dim * jnp.outer(jnp.ones(dim), jnp.ones(dim)) * A[0,0] \
                    + 1/jnp.sqrt(dim) * jnp.outer(jnp.ones(dim), A[0, 1:] @ U_matrix) + 1/jnp.sqrt(dim) * U_matrix.T @ jnp.outer(A[1:, 0], jnp.ones(dim))
        
        return jnp.linalg.norm(jnp.diag(val_M) - jnp.ones(dim) * jnp.trace(A)/dim) 
    
    l_diff_eps = 1e-5
    g_l = grad(l)
    linesearch = orthog_linesearch(l, c1=0.1, c2=0.9)

    l_hist = []
    for _ in range(num_iter):
        if len(l_hist) > 1 and abs(l_hist[-2][0][1] - l_hist[-1][0][1]) < l_diff_eps:
            break

        G = g_l(X.flatten()).reshape(dim - 1, dim - 1)
        U = P.T @ convert_to_U(X, 0)
        S = V @ sing_vals @ U
        S = U_D @ S
        jrandom_key, subkey = jrandom.split(jrandom_key)
        l_hist.append(loss(S, subkey))
        
        A_X = G @ X.T - X @ G.T

        Y = lambda tau: jnp.linalg.inv(I + tau/2 * A_X) @ (I - tau/2 * A_X) @ X
        
        alpha = linesearch(X, Y, A_X)

        X = Y(alpha)


    return X, l_hist



def create_S(H, sig, coeff, num_iter, jrandom_key, loss):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = jnp.linalg.eigh(H)
    U_D = jnp.real(U_D)
    D = jnp.real(jnp.diag(D))

    D_diag = jnp.diag(D)

    sing_vals, _ = generate_sing_vals_V(D_diag, sig, coeff)
    lmbda = jnp.diag(sing_vals)**2
    P = np.eye(dim)
    tmp_row = P[lmbda.argmax()].copy()
    P[lmbda.argmax()] = P[0].copy()
    P[0] = tmp_row
    P = jnp.array(P)

    V = jnp.eye(dim)
    
    A = sing_vals @ V.T @ D @ V @ sing_vals
    W, l_hist = optimize_W(P @ A @ P.T, P, sing_vals, V, U_D, num_iter, jrandom_key, loss, x_init=None) 
    U = P.T @ convert_to_U(W, 0)
    S = V @ sing_vals @ U
    S = U_D @ S
    return S, l_hist

def get_sampling_set_loss(H, sig, coeff, jrandom_key, loss, num_iter):
    return create_S(H, sig, coeff, num_iter, jrandom_key, loss)[1]