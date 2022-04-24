
from filecmp import dircmp
import numpy as np
import jax.numpy as jnp
from jax import grad
import jax.random as jrandom

from generate_sing_vals_V import generate_sing_vals_V

from jax.config import config
config.update("jax_enable_x64", True)

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

def orthog_linesearch(l, c1, c2):

    def helper(X, search_direction, A):
        f0 = l(X)
        g_tau_0 = -1/2 * jnp.linalg.norm(A, "fro")**2
        
        def armijo_rule(alpha):
            return (l(search_direction(alpha)) > f0 + c1*alpha*g_tau_0) # and alpha > 0.001
        
        def armijo_update(alpha):
            return c2*alpha
            
        alpha = 10
        max_calls = 1000
        while armijo_rule(alpha) and max_calls > 0:
            alpha = armijo_update(alpha)
            max_calls -= 1

        return alpha

    return helper


def optimize_W(A, num_iter, x_init=None, jrandom_key=None):
    """Constraint is U.c = \bar{c} 1 and U.U^T = I"""
    
    dim = len(A)
    
    R = generate_regular_simplex(dim - 1)
    
    # init X
    if x_init is None:
        X = jnp.eye(dim - 1)
        # if jrandom_key is not None:
        #     jrandom_key, subkey = jrandom.split(jrandom_key)
        #     X = jrandom.normal(subkey, shape=(dim-1, dim-1))
        #     X = jnp.real(jnp.linalg.eigh(X + X.T)[1])
    else:
        X = x_init
    I = jnp.eye(dim - 1)
    
    
    def l(U):
        U_matrix = jnp.sqrt((dim - 1)/dim) * U.reshape(dim - 1, dim - 1) @ R
        val_M = U_matrix.T @ A[1:, 1:] @ U_matrix + 1/dim * jnp.outer(jnp.ones(dim), jnp.ones(dim)) * A[0,0] \
                    + 1/jnp.sqrt(dim) * jnp.outer(jnp.ones(dim), A[0, 1:] @ U_matrix) + 1/jnp.sqrt(dim) * U_matrix.T @ jnp.outer(A[1:, 0], jnp.ones(dim))
        
        return jnp.linalg.norm(jnp.diag(val_M) - jnp.ones(dim) * jnp.trace(A)/dim) 
        # return jnp.linalg.norm(jnp.diag(V.T @ U_matrix.T @ jnp.diag(c) @ U_matrix @ V) - jnp.ones(dim + 1) * jnp.mean(c)) 
    
    
    g_l = grad(l)
    linesearch = orthog_linesearch(l, c1=0.1, c2=0.9)


    eps = 1e-6
    l_diff_eps = 1e-8

    l_hist = []
    for _ in range(num_iter):
        # print(_)
        
        G = g_l(X.flatten()).reshape(dim - 1, dim - 1)
        l_hist.append(l(X))
        
        # if l_hist[-1] < l_eps:
            # break

        # if jnp.linalg.norm(G) < eps:
            # break

        if len(l_hist) > 1 and (l_hist[-2] - l_hist[-1]) < l_diff_eps:
            break
        

        A_X = G @ X.T - X @ G.T
        # P_X = I - 1/2 * X @ X.T
        # A_X = P_X @ G @ X.T - X @ (P_X @ G).T


        Y = lambda tau: jnp.linalg.inv(I + tau/2 * A_X) @ (I - tau/2 * A_X) @ X
        
        alpha = linesearch(X, Y, A_X)

        X = Y(alpha)


    return X, l_hist



def create_S(H, sig, num_iter=10, x_init=None, coeff=0.1):
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
    W, l_hist = optimize_W(P @ A @ P.T, num_iter, x_init=x_init) 
    U = P.T @ convert_to_U(W, 0)
    S = V @ sing_vals @ U
    return U_D @ S, W
