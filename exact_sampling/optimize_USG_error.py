import jax.numpy as jnp
import jax.random as jrandom

import matplotlib.pyplot as plt
import pandas as pd
from jax import jit, grad, jacfwd, jacrev, xla_computation

from Functions import Quadratic, Ackley, Brown 

from scipy.stats import linregress

from tqdm import tqdm
import pickle

import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)

def get_a(D_min, sum_sqrt, dim, sig):
    disc = sig**4 * (sum_sqrt**2) * (8 * (dim + 1) * D_min + sum_sqrt**2)
    return 1/jnp.sqrt(dim * D_min) * jnp.sqrt(jnp.sqrt(disc) + 2 * (dim + 1) * D_min * sig**2 + sig**2 * sum_sqrt**2)

def get_lambda_max(D_min, sum_sqrt, dim, sig):
    disc = sig**4 * (sum_sqrt**2) * (8 * (dim + 1) * D_min + sum_sqrt**2)
    numerator = jnp.sqrt(disc) + 4 * (dim + 1) * D_min * sig**2 + sig**2 * sum_sqrt**2
    denom = D_min**(1.5) * 2 * jnp.sqrt(1/dim) * jnp.sqrt(jnp.sqrt(disc) + 2 * (dim + 1) * D_min * sig**2 + sig**2 * sum_sqrt**2)
    return numerator/denom

def get_full_sing_vals(D, dim, sig):
    D_diag = jnp.abs(jnp.diag(D))
    max_row = jnp.argmin(D_diag)
    sum_sqrt = jnp.sum(jnp.sqrt(D_diag)) - jnp.sqrt(D_diag[max_row])
    lmbda_max = get_lambda_max(jnp.min(D_diag), sum_sqrt, dim, sig)
    a = get_a(jnp.min(D_diag), sum_sqrt, dim, sig)
    lmbda = sig*np.array(jnp.sqrt(lmbda_max / (a * D_diag)))
    lmbda = lmbda.at[max_row].set(lmbda_max)
    lmbda = jnp.array(lmbda)
    sing_val = jnp.sqrt(lmbda)
    return sing_val

def loss_getter(dim, N, H, sig):
    def helper(X):

        S = X.reshape(N, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ H @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        
        return 1/2 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term)

    return helper

def createU_4(max_row):
    
    res = np.array([[ 1.,  1.,  1.,  1.],
                     [ 1.,  1., -1., -1.],
                     [ 1., -1.,  1., -1.],
                     [-1.,  1.,  1., -1.]])
    
    tmp_row = res[max_row].copy()
    res[max_row] = jnp.ones(4)
    res[0] = tmp_row
    return 1/jnp.sqrt(4) * jnp.array(res)

def generate_regular_simplex(dim):
    res = []
    I = np.eye(dim)
    for i in range(dim):
        res.append(jnp.sqrt(1 + 1/dim) * I[i] - 1/pow(dim, 3/2) *(np.sqrt(dim + 1) + 1) * np.ones(dim))
        
    res.append(1/np.sqrt(dim) * np.ones(dim))
    
    return jnp.array(res).T

def convert_to_U(W, to_insert):
    dim = W.shape[0] + 1
    V = generate_regular_simplex(dim - 1)
    tmp_U = (jnp.sqrt((dim - 1)/dim) * W @ V)
    U = jnp.insert(tmp_U, to_insert, jnp.ones(shape=(1, dim))/jnp.sqrt(dim), axis=0)
    return U

def construct_c(sing_vals, D):
    return sing_vals**2 * jnp.diag(D)

def orthog_linesearch(l, c1, c2):

    def helper(X, search_direction, A):
        f0 = l(X)
        g_tau_0 = -1/2 * jnp.linalg.norm(A, "fro")**2
        
        def armijo_rule(alpha):
            return (l(search_direction(alpha)) > f0 + c1*alpha*g_tau_0) # and alpha > 0.001
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        max_calls = 100
        while armijo_rule(alpha) and max_calls > 0:
            alpha = armijo_update(alpha)
            max_calls -= 1

        return alpha

    return helper


def optimize_W(c, num_iter, x_init=None):
    """Constraint is U.c = \bar{c} 1 and U.U^T = I"""
    
    dim = len(c)
    
    V = generate_regular_simplex(dim)
    
    # init X
    if x_init is None:
        X = jnp.eye(dim)
    else:
        X = x_init
    I = jnp.eye(dim)
    
    
    def l(U):
        U_matrix = U.reshape(dim, dim)
        return jnp.linalg.norm(jnp.diag(V.T @ U_matrix.T @ jnp.diag(c) @ U_matrix @ V) - jnp.ones(dim + 1) * jnp.mean(c)) 
    
    
    g_l = grad(l)
    linesearch = orthog_linesearch(l, c1=0.1, c2=0.9)


    eps = 1e-4
    l_eps = 1e-2
    
    l_hist = []
    for _ in range(num_iter):
        num_iter -= 1
        
        G = g_l(X.flatten()).reshape(dim, dim)
        l_hist.append(l(X))
        
        if l_hist[-1] < l_eps:
            break

        if jnp.linalg.norm(G) < eps:
            break
        
        A = G @ X.T - X @ G.T
                
        Y = lambda tau: jnp.linalg.inv(I + tau/2 * A) @ (I - tau/2 * A) @ X
        
        alpha = linesearch(X, Y, A)

        
        X = Y(alpha)
    # plt.plot(l_hist)
    # plt.show()
    print(l_hist[-1])
    return X, l_hist


def create_S(H, sig, num_iter=10, x_init=None):
    dim = H.shape[0] 

    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_D = jnp.linalg.eig(H)
    U_D = jnp.real(U_D)
    D = jnp.abs(jnp.real(jnp.diag(D)))
    # print(repr(jnp.diag(D)))
    # D = D + 0.01 * jnp.eye(dim)
    # print(D)

    sing_vals = get_full_sing_vals(D, dim, sig)
    c = construct_c(sing_vals, D)
    min_D = jnp.diag(D).argmin()
    W, l_hist = optimize_W(np.delete(c, min_D), num_iter, x_init=x_init) 
    U = convert_to_U(W, min_D)
    S = jnp.diag(sing_vals) @ U

    return U_D @ S, W



if __name__ == "__main__":

    dim = 3
    # D = np.linspace(1, 100, dim)
    # np.random.shuffle(D)
    # D = np.diag(D)
    # D = jnp.diag(jnp.array([  50.5, 75.25,25.75, 100, 1000]))
    sig = 0.1

    l = loss_getter(dim, dim, D, sig)

    S, _ = create_S(D, sig, num_iter=50)
    print(l(S.T.flatten()))




