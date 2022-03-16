"""Here we define the barrier functions and at the same time the domain."""

import numpy as np
import time
import jax.numpy as jnp
import jax
from functools import partial
from jax import jit
import jax.random as jrandom

def get_barrier(config, jrandom_key):
    prob_config = config["Optimization_Problem"]
    if prob_config["name"] == "Linear":
        num_barriers = prob_config["num_barriers"]
        dim = config["dim"]

        jrandom_key, subkey = jrandom.split(jrandom_key)
        dirs = jrandom.normal(subkey, shape=(num_barriers, dim)) # sample gaussian and normalize 
        ws = dirs/np.linalg.norm(dirs, axis=1).reshape(-1, 1)

        jrandom_key, subkey = jrandom.split(jrandom_key)
        A = jrandom.normal(subkey, shape=(dim, dim))
        U, _, _ = jnp.linalg.svd(A) # some rotation matrix
        S = jnp.diag(jnp.linspace(0.1, 5, dim)) # the axis scaling
        ws = ws.dot(S.dot(U.T))
        bs = np.ones(num_barriers)
        noise_std = config["Optimization_Problem"]["barrier_noise"]
        print(repr(ws))
        if noise_std is None:
            noise_std = 0
        B = LogPolytopeBarrier(ws, bs, noise_std) 
    else: 
        raise ValueError("Does not support given barrier type {} with domain {}".format(config["optimization_meta"]["barrier_type"], config["domain_name"]))
    return B

class LogPolytopeBarrier:

    def __init__(self, ws, bs, noise_std=0):
        """ws.shape = (N, d), bs.shape = (N)"""
        self.ws = jnp.array(ws)
        self.bs = jnp.array(bs)
        self.dim = len(ws[0])
        self.noise_std = noise_std

    # @partial(jit, static_argnums=(0,))
    def _get_dists(self, xs):
        """We consider the sum of log barrier (equivalent to considering each barrier to be a potential function).
        Distance to a hyperplane w.x = b is given by | w.x/|w| - b/|w| |. We consider the absolute value of this, which follows the assumption that if we are on the a side of the hyperplane we stay there. 
        However, the signs tell us whether we are on the side of the hyperplane which is closer to the origin. If the sign is negative, then we are closer."""
        
        xs_len_along_ws = xs.dot(self.ws.T)/jnp.linalg.norm(self.ws, axis=1)
        hyperplane_dist = self.bs/jnp.linalg.norm(self.ws, axis=1)
        dists = xs_len_along_ws - hyperplane_dist # dists.shape = (N_x, N_ws)
        signs = 2*(dists * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dists), signs
    
    # @partial(jit, static_argnums=(0,))
    def dir_dists(self, xs, dirs):
        # we get the distance of the direction to every boundary (if parallel we have infty). We have w.(x0 + td) = b. Hence, t = (b - w.x0)/(w.d). So t is the scale to apply to d to get to the hyperplane. 
        xs_len_along_ws = xs.dot(self.ws.T)/(dirs.dot(self.ws.T))
        hyperplane_dist = self.bs/(dirs.dot(self.ws.T))
        dists = xs_len_along_ws - hyperplane_dist # dists.shape = (N_x, N_ws)
        signs = 2*(dists * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dists), signs
    
    # @partial(jit, static_argnums=(0,))
    def f(self, xs, jrandom_key=None):
        """x.shape = (N, d). Outside of the bounded region around zero we are infinite."""
        dists, signs = self._get_dists(xs) 
        ret = -jnp.sum(jnp.log(dists), axis=1) # shape = (N_x)
        ret = jnp.where(jnp.any(signs > 0, axis=1), jnp.inf, ret)
        if jrandom_key is not None:
            return ret + self.noise_std * jrandom.normal(jrandom_key, shape=(xs.shape[0],)) 
        return ret

    # @partial(jit, static_argnums=(0,))
    def f1(self, xs):
        dists, signs = self._get_dists(xs)
        grads = (1/dists * signs).dot((-self.ws / jnp.linalg.norm(self.ws, axis=1).T.reshape(-1, 1)))
        return grads

    # @partial(jit, static_argnums=(0,))
    def f2(self, xs):
        normalized_ws = self.ws / jnp.linalg.norm(self.ws, axis=1).reshape(-1, 1)
        dists, signs = self._get_dists(xs)
        hess = []
        for i in range(len(xs)):
            hess.append(jnp.dot(normalized_ws.T, 1/(dists[i].reshape(-1, 1))**2 * normalized_ws))

        hess = jnp.array(hess)
        return hess 
    
    # @partial(jit, static_argnums=(0,))
    def f2_inv(self, x):
        f2 = self.f2(x)
        return jnp.array([jnp.linalg.inv(f2[i]) for i in range(len(f2))])


        

class EntropicBarrier():

    def __init__(self):
        pass