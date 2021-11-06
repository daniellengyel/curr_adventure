from jax import random as jrandom
import jax.numpy as jnp
from .grid_generation import grid_generator

def SD_1E0(F, x_0, sampling_config, jrandom_key, sample_based=True):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, cov = grid_generator(x_0, sampling_config, subkey)  
    if sample_based: 
        ru = sample_points - jnp.mean(sample_points, axis=0)
        cov = jnp.cov(sample_points.T, bias=True).reshape(x_0.shape[0], x_0.shape[0])
    else:
        ru = sample_points - x_0
    jrandom_key, subkey = jrandom.split(jrandom_key)
    out_vals = F.f(sample_points, subkey)
    f_ru = out_vals.T.dot(ru)/sample_points.shape[0]
    return jnp.linalg.inv(cov).dot(f_ru)


def FD_1E0(F, x_0, h, num_dir_samples, jrandom_key):
    dim = x_0.shape[0]

    forwardX = h*jnp.eye(dim) + x_0
    backwardX = -h*jnp.eye(dim) + x_0

    X = jnp.concatenate([jnp.tile(forwardX, (num_dir_samples, 1)), jnp.tile(backwardX, (num_dir_samples, 1))])
    F_vals = F.f(X, jrandom_key)
    G = F_vals.T.dot(X - x_0) / (2 * h**2 * num_dir_samples) 

    return G

