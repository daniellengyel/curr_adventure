
import numpy as np
import jax.numpy as jnp
from jax import grad
import jax.random as jrandom

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
        return 1/4 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S, ord="fro")**4

    return helper

def optimize_uncentered_S(H, sig, coeff, max_steps=15, jrandom_key=None, x_curr=None):
    
    dim = H.shape[0]
    N = dim
    
    H = (H + H.T) / 2. # to combat numerical inaccuracies. 
    D, U = jnp.linalg.eigh(H)
    U = jnp.real(U)

    D = jnp.diag(D)

    if x_curr is None:

        if jrandom_key is None:
            x_curr = jnp.diag(jnp.sqrt(sig / jnp.abs(jnp.diag(D)))).flatten()
            pass
        else:
            x_curr = jrandom.normal(jrandom_key, shape=(dim * dim,)) * 0.1
    
    l = loss_getter(dim, N, D, sig, coeff)
    g_l = grad(l)
    linesearch = helper_linesearch(l, g_l, c1=0.1, c2=0.9)

    eps = 1e-20

    for t in range(max_steps):

        curr_grad = g_l(x_curr)
    
        search_direction = -curr_grad

        if jnp.linalg.norm(curr_grad) < eps:
            break    

        alpha = linesearch(x_curr, search_direction)

        x_curr += alpha*search_direction 
    
    return U.dot(x_curr.reshape(N, dim).T)