from jax import random as jrandom
import jax.numpy as jnp
import jax
from jax import jit, partial

"""Assume mu.shape = (d) and cov.shape = (d, d)"""

def grid_generator(x_0, config, jrandom_key): 
    jrandom_key, subkey = jrandom.split(jrandom_key)
    
    dim = len(x_0)
    N = config["N"]
    h = config["h"]
    cov = config["cov"]
    is_uniform_sphere_random = config["is_uniform_sphere_random"]

    dirs = generate_ellipse(dim, N, cov, is_uniform_sphere_random, subkey)

    jrandom_key, subkey = jrandom.split(jrandom_key)


    rs = rescale_ellipse(N, "exact", h, subkey)
    # if config["distribution_name"] == "beta":
    #     if config["ensure_within_domain"]:
    #         dists = config["F"].dir_dists(x_0, dirs) # for each dir get distance to boundary
    #         R = jnp.min(dists[0])
    #     else:
    #         R = 1
    #     alpha = beta_alpha(h, R)
    #     rs = R*(jrandom.beta(subkey, alpha, alpha, shape=(N, 1)) - 0.5) * 2
    grid_points = dirs * rs

    return x_0 + jnp.array(grid_points), cov*h**2

def generate_ellipse(dim, N, ellipse_M, is_uniform_sphere_random, jrandom_key): 

    # generate points on the sphere
    if is_uniform_sphere_random:
        dirs = jrandom.normal(jrandom_key, shape=(N, dim)) 
        dirs = dirs/jnp.linalg.norm(dirs, axis=1).reshape(-1, 1) * jnp.sqrt(dim)
    else:
        forwardX = jnp.eye(dim)
        backwardX = -jnp.eye(dim)

        num_dir_samples = N // (2 * dim)
        if num_dir_samples == 0:
            raise Exception("Given too few samples, {}, for Central Differences with dimension {}".format(N, dim))

        dirs = jnp.concatenate([jnp.tile(forwardX, (num_dir_samples, 1)), jnp.tile(backwardX, (num_dir_samples, 1))])

    # transform sphere to ellipse
    if ellipse_M is not None:
        L = jnp.linalg.cholesky(ellipse_M)
        dirs = dirs.dot(L.T)

    return dirs 

def rescale_ellipse(N, rescale_method, h, jrandom_key):
    if rescale_method == "exact":
        return h * jnp.ones(shape=(N, 1))

def beta_alpha(h, R=1):
    return 1/2. * (R**2/h**2 - 1)


def beta_covariance(dim, R, alpha):
    """Returns scalar sigma, since the covariance matrix is sigma*I_{dim}. So we are saving space."""
    return (R**2)/(1 + 2 * alpha) * 1/dim

