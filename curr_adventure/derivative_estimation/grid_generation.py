from jax import random as jrandom
import jax.numpy as jnp
import jax
from jax import jit, partial
from .adaptive_FD_interlval_estimation import adaptive_interval, CD_testing_ratio

"""Assume mu.shape = (d) and ellipse_M.shape = (d, d)"""

def grid_generator(x_0, config, jrandom_key): 
    jrandom_key, subkey = jrandom.split(jrandom_key)
    
    dim = len(x_0)
    N = config["N"]
    h = config["h"]
    if "costum" in config and config["costum"] is not None:
        return config["costum"]
    if "h_adaptive" in config and config["h_adaptive"]:
        return adaptive_grid(x_0, config, jrandom_key)

    ellipse_M = config["ellipse_M"]
    if ellipse_M is None:
        ellipse_M = jnp.eye(x_0.shape[0])
    is_uniform_sphere_random = config["is_uniform_sphere_random"]

    dirs = generate_ellipse(dim, N, ellipse_M, is_uniform_sphere_random, subkey)

    jrandom_key, subkey = jrandom.split(jrandom_key)

    rs = rescale_ellipse(N, "exact", h, subkey)
    # if config["distribution_name"] == "beta":
    #     if config["ensure_within_domain"]:
    #         
    #     else:
    #         R = 1
    #     alpha = beta_alpha(h, R)
    #     rs = R*(jrandom.beta(subkey, alpha, alpha, shape=(N, 1)) - 0.5) * 2
    dists = config["F"].dir_dists(x_0, dirs) # for each dir get the scale to apply to dir to reach boundary
    R = jnp.min(dists[0])
    rs = R * rs
    grid_points = dirs * rs

    return x_0 + jnp.array(grid_points), ellipse_M*h**2

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
        # D, U = jnp.linalg.eigh(ellipse_M)
        # L = U.dot(jnp.diag(jnp.sqrt(D)))
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


def adaptive_grid(x_0, config, jrandom_key, last_hs=None):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    
    dim = len(x_0)
    N = config["N"]
    sigma_f = config["sigma_f"]

    forwardX = jnp.eye(dim)
    backwardX = -jnp.eye(dim)

    num_dir_samples = N // (2 * dim)
    if num_dir_samples == 0:
        raise Exception("Given too few samples, {}, for Central Differences with dimension {}".format(N, dim))


    r_star = 3
    beta = 2
    eta = 0.1
    q = 3
    r_l = max(1 + eta, r_star/beta)
    r_u = max(3 * (1 + eta), beta * r_star)

    hs = []
    F = config["F"]
    for i, p in enumerate(jnp.eye(dim)):
        if last_hs is None:
            h_0 = sigma_f**(1./q)
        else:
            h_0 = last_hs[i]
        testing_ratio = CD_testing_ratio(F, x_0, p, sigma_f)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        hs.append(adaptive_interval(testing_ratio, r_l, r_u, h_0, eta, jrandom_key))

    hs = jnp.array(hs)

    forwardX = jnp.eye(dim) * hs
    backwardX = -jnp.eye(dim) * hs

    dirs = jnp.concatenate([jnp.tile(forwardX, (num_dir_samples, 1)), jnp.tile(backwardX, (num_dir_samples, 1))])


    return dirs