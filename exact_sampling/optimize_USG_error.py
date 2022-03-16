import jax.numpy as jnp
import jax.random as jrandom

import matplotlib.pyplot as plt
import pandas as pd
from jax import jit, grad, jacfwd, jacrev

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

def optimize_uncentered_S(H, sig, max_steps=15):
    
    dim = H.shape[0]
    N = dim
    
    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U = jnp.linalg.eig(H)
    U = jnp.real(U)

    # print("D", repr(jnp.real(D)))
    D = jnp.diag(D)

    x_curr = jnp.diag(jnp.sqrt(sig / jnp.abs(jnp.diag(D)))).flatten()
    
    l = loss_getter(dim, N, D, sig)
    g_l = grad(l)
    linesearch = helper_linesearch(l, g_l, c1=0.01, c2=0.7)

    eps = 1e-1

    res = []
    res_x = []

    for t in range(max_steps):

        res.append(l(x_curr))
#         res_x.append(x_curr) # TURN ON FOR HISTORY 

        curr_grad = g_l(x_curr)
        # print(curr_grad)
        if jnp.linalg.norm(curr_grad) < eps:
            break    

        search_direction = -curr_grad
        alpha = linesearch(x_curr, search_direction)
        # print(x_curr)
        S = x_curr.reshape(N, dim).T

        SS = S.dot(S.T)
        # print(SS)
        # print(jnp.linalg.eig(SS)[0])

        # print(alpha)
        x_curr += alpha*search_direction #+ 1/np.log(float(t + 2)) * np.random.normal(size=x_curr.shape)
    
    return U.dot(x_curr.reshape(N, dim).T), res


if __name__ == "__main__":


    def simplex_gradient(F, x_0, S, subkey_f):
        jrandom_key, subkey = jrandom.split(subkey_f)
        FS = F.f(S.T + x_0, subkey)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        F_x_0 = F.f(x_0.reshape(1, -1), subkey)
        SS_inv = jnp.linalg.inv(S.dot(S.T))
        return SS_inv.dot(S.dot(FS - F_x_0))

        
    # jrandom_key = jrandom.PRNGKey(0)
    # jrandom_key, subkey = jrandom.split(jrandom_key)
    # sig = 0.1
    # F = Brown(sig)
    # dim = 50
    # max_steps_USG = 10

    # x_curr = jrandom.normal(subkey, shape=(dim,)) * 0.2

    # jrandom_key, subkey = jrandom.split(jrandom_key)

    # H = F.f2(x_curr)
    
    # jrandom_key, subkey = jrandom.split(jrandom_key)
    # S, _ = optimize_uncentered_S(H, sig, max_steps=max_steps_USG)

    # # print(simplex_gradient(F, x_curr, S, subkey))
    # # print(F.f1(x_curr))

    # H_err = jrandom.normal(subkey, H.shape) * 0.1 # TODO with noise. 
    # H += H_err.T.dot(H_err)
    # jrandom_key, subkey = jrandom.split(jrandom_key)
    # S, _ = optimize_uncentered_S(H, sig, max_steps=max_steps_USG)

    # # print(simplex_gradient(F, x_curr, S, subkey))
    # # print(F.f1(x_curr))

    D = jnp.array([74.32926   ,  0.94854367,  1.0000002 ,  1.0000013 ,
              1.0000001 ,  1.0000001 ,  0.9999995 ,  0.9999995 ,
              1.0000011 ,  1.0000011 ,  1.0000005 ,  0.99999905,
              0.99999905,  0.9999993 ,  0.9999993 ,  0.99999964,
              0.99999964,  0.9999988 ,  0.99999964,  0.99999964,
              0.99999917,  0.99999934,  0.9999997 ,  1.0000001 ,
              0.99999946,  0.99999976,  1.0000005 ,  1.0000007 ,
              0.9999994 ,  0.9999996 ,  1.0000004 ,  0.99999964,
              0.99999994,  0.99999994,  0.9999997 ,  1.0000001 ,
              1.        ,  0.9999999 ,  0.99999964,  0.9999997 ,
              1.0000001 ,  0.9999999 ,  0.99999964,  0.9999998 ,
              1.        ,  0.99999976,  0.9999996 ,  0.99999976,
              0.99999994,  1.        ])
    D = jnp.diag(D)

    with open("./tmp.pkl", "rb") as f:
        H = jnp.array(pickle.load(f))


    jrandom_key = jnp.array([3768195656, 3102107202], dtype="uint32")
    sig = 0.01
    F = Brown(sig)    
    x_0 = jnp.array([-1.04332134e-01, -1.28196761e-01, -4.17667553e-02,
             -1.89537667e-02,  6.60059825e-02, -4.99424338e-02,
              1.29436031e-01, -8.28816146e-02, -2.19505727e-02,
              3.86416316e-02, -1.16837084e-01,  9.48305130e-02,
             -4.58742231e-02, -2.53489837e-02, -3.49037126e-02,
             -2.19875276e-02, -3.06242779e-02, -1.80370435e-02,
              3.02513391e-02, -5.40898666e-02, -1.11921221e-01,
             -3.85048613e-02, -7.20525086e-02, -3.59889045e-02,
             -1.67415440e-02,  1.79341093e-01, -9.31865573e-02,
              2.07182407e-01, -3.86191010e-02,  1.01104118e-01,
             -4.00085449e-02,  3.09279189e-04, -9.90203023e-03,
             -2.49819458e-03,  7.19085336e-04,  7.14532286e-03,
             -2.84700990e-02, -4.98940349e-02, -1.46475747e-01,
             -5.93610890e-02, -4.56139818e-02, -7.63684511e-05,
             -9.13031586e-03, -1.40764415e-02,  1.57458246e-01,
             -4.36275899e-02,  9.23807770e-02, -5.08539677e-02,
             -5.73905706e-02,  3.31046470e-02])

# [ -3.2192605   8.         -4.25       -1.75       11.5        17.75
#   13.75       11.         18.        -10.1875     13.5         4.0625
#    7.         -1.125      -1.         -4.25        3.         -3.75
#   -4.5         4.5         0.          5.25      -21.        -10.
#    2.875       1.5         0.          0.4375      0.5         4.75
#  -14.        -11.          4.5         2.          7.          1.25
#   -3.25       -1.875       6.          3.         -2.75        0.
#   -1.25       -2.          2.5        -1.125       1.125       0.5
#    0.75       -0.5      ]
    H = jnp.array(H, dtype=np.float64)

    D, U = jnp.linalg.eig(H)
    # print(U.T.dot(U))

    S, _ = optimize_uncentered_S(H, sig, max_steps=10)
    # print(_)
    sg = simplex_gradient(F, x_0, S, jrandom_key)


    # print(jnp.linalg.eig(H)[0])
    # print(jnp.linalg.eig(F.f2(x_0))[0])
    # print(jnp.linalg.norm(H - F.f2(x_0))/jnp.linalg.norm(F.f2(x_0)))

    # print("sg", sg)
    # print("true", F.f1(x_0))
    print("diff", jnp.linalg.norm(F.f1(x_0) - sg)/jnp.linalg.norm(F.f1(x_0)))

    h = 0.1
    print("Diff fd with h {}".format(h),  jnp.linalg.norm(F.f1(x_0) - simplex_gradient(F, x_0, h*jnp.eye(x_0.shape[0]), jrandom_key)))

    # D = jnp.array([[1.9999988+0.j, 0.       +0.j, 0.       +0.j, 0.       +0.j, 0.       +0.j],
    #                [0.       +0.j, 3.995916 +0.j, 0.       +0.j, 0.       +0.j, 0.       +0.j],
    #                [0.       +0.j, 0.       +0.j, 3.9934745+0.j, 0.       +0.j, 0.       +0.j],
    #                [0.       +0.j, 0.       +0.j, 0.       +0.j, 3.9971404+0.j, 0.       +0.j],
    #                [0.       +0.j, 0.       +0.j, 0.       +0.j, 0.       +0.j, 1.995689 +0.j]])
    # # D = jnp.array([[2., 0.       +0.j, 0.       +0.j, 0.       +0.j, 0.       +0.j],
    # #                [0.       +0.j, 4., 0.       +0.j, 0.       +0.j, 0.       +0.j],
    # #                [0.       +0.j, 0.       +0.j, 4., 0.       +0.j, 0.       +0.j],
    # #                [0.       +0.j, 0.       +0.j, 0.       +0.j, 4., 0.       +0.j],
    # #                [0.       +0.j, 0.       +0.j, 0.       +0.j, 0.       +0.j, 1.9995]])
    
    # sig = 0.001




