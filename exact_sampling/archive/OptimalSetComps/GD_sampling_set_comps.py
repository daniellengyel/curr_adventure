import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jax import grad


import scipy 

import matplotlib.pyplot as plt


from tqdm import tqdm
import time

from jax.config import config
config.update("jax_enable_x64", True)




def helper_linesearch(f, g, c1, c2):

    def helper(X, search_direction):
        f0 = f(X)
        f1 = g(X)
        dg = jnp.inner(search_direction, f1)

        def armijo_rule(alpha):
            return f(X + alpha * search_direction) > f0 + c1*alpha*dg
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 1
        while armijo_rule(alpha):
            alpha = armijo_update(alpha)


        return alpha

    return helper


def loss_getter(dim, N, H, sig, coeff):
    def helper(X):

        S = X.reshape(N, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ H @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        return 1/2 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S)**4

    return helper

def optimize_uncentered_S(H, sig, coeff, jrandom_key, loss_tracker, max_steps=15):
    dim = H.shape[0]
    N = dim
    
    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U_H = jnp.linalg.eigh(H)
    U_H = jnp.real(U_H)
    D = jnp.diag(D)

    x_curr = jnp.diag(2 * jnp.sqrt(sig / jnp.abs(jnp.diag(D)))).flatten()
            
    
    l = loss_getter(dim, N, D, sig, coeff)
    g_l = grad(l)
    linesearch = helper_linesearch(l, g_l, c1=0.1, c2=0.9)

    eps = 1e-20

    res = []

    for t in range(max_steps):

        jrandom_key, subkey = jrandom.split(jrandom_key)
        res.append(loss_tracker(U_H @ x_curr.reshape(dim, dim).T, subkey))

        curr_grad = g_l(x_curr)
    
        search_direction = -curr_grad

        if jnp.linalg.norm(curr_grad) < eps:
            break    

        alpha = linesearch(x_curr, search_direction)


        S = x_curr.reshape(N, dim).T

        SS = S.dot(S.T)

        x_curr += alpha*search_direction 


    return res


if __name__ == "__main__":

    dim = 10
    H = jnp.diag(jnp.linspace(-2, 10, dim))

    sig = 0.1
    coeff = 0.1



    _, l_hist = optimize_uncentered_S(H, sig, coeff, max_steps=15, jrandom_key=None, x_curr=None) 

    plt.plot(l_hist[:, 1], l_hist[:, 0])
    plt.show()
