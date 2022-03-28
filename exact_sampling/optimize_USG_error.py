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


def loss_getter_orthog(dim, N, H, sig):
    def helper(X):
    
        S = jnp.diag(X)

        SS = S.dot(S.T)

        SS_inv = jnp.linalg.inv(SS)

        first_term = S.dot(jnp.diag(S.T.dot(H.dot(S))))
        
        second_term = jnp.linalg.norm(SS_inv.dot(S), ord="fro")**2    
        S_sum = jnp.sum(S, axis=1)  
        third_term = S_sum.T.dot(SS_inv.dot(SS_inv.dot(S_sum)))  
        
        return jnp.linalg.norm(SS_inv.dot(first_term))**2 + sig**2*(second_term + third_term)
    return helper


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

def loss_getter(dim, N, H, sig):
    def helper(X):
    
        S = X.reshape(N, dim).T

        SS = S.dot(S.T)

        SS_inv = jnp.linalg.inv(SS)
        S_sum = jnp.sum(S, axis=1)  

        first_term = S.dot(jnp.diag(S.T.dot(H.dot(S))))
        second_term = jnp.linalg.norm(SS_inv.dot(S), ord="fro")**2        
        third_term = S_sum.T.dot(SS_inv.dot(SS_inv.dot(S_sum)))  
        
        return jnp.linalg.norm(SS_inv.dot(first_term))**2 + sig**2*(second_term + third_term)
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




def optimize_uncentered_S(H, sig, max_steps=15, prev_rot_S = None):
    
    dim = H.shape[0]
    N = dim

    assert dim == 4
    
    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U = jnp.linalg.eig(H)
    U = jnp.real(U)

    D = jnp.abs(jnp.diag(D))

    SU = createU_4(jnp.argmin(jnp.diag(D)))
    SD = get_full_sing_vals(D, dim, sig)

    S = jnp.diag(SD) @ SU

#     print("D", repr(jnp.real(D)))

#     if prev_rot_S is None:
#         x_curr = jnp.diag(jnp.sqrt(sig / jnp.abs(jnp.diag(D)))).flatten()
#     else:
#         x_curr = prev_rot_S.T.flatten()
    
#     l = loss_getter(dim, N, D, sig)
#     g_l = grad(l)
#     linesearch = helper_linesearch(l, g_l, c1=0.01, c2=0.7)

#     eps = 1e-9

#     res = []
#     res_x = []

#     # print(D)
#     # print(sig)

#     for t in range(max_steps):

#         res.append(l(x_curr))
#         # print(res[-1])
# #         res_x.append(x_curr) # TURN ON FOR HISTORY 

#         curr_grad = g_l(x_curr)
#         if jnp.linalg.norm(curr_grad) < eps:
#             break    

#         search_direction = -curr_grad
#         alpha = linesearch(x_curr, search_direction)
#         S = x_curr.reshape(N, dim).T

#         SS = S.dot(S.T)

#         x_curr += alpha*search_direction
#     # print(x_curr.reshape(N, dim).T)
    return U.dot(S), [], S


if __name__ == "__main__":


    pass




