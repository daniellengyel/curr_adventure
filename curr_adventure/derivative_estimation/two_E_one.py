from jax import random as jrandom
import jax.numpy as jnp

def SD_2E1(F, x_0, sampling_config, jrandom_key, sample_based=True):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, cov = jax_hit_run(x_0, sampling_config, subkey)  
    if sample_based: 
        ru = sample_points - jnp.mean(sample_points, axis=0)
    else:
        ru = sample_points - x_0
    jrandom_key, subkey = jrandom.split(jrandom_key)
    out_grads = F.f1(sample_points, subkey)
    g_ru = out_grads.T.dot(ru)/sample_points.shape[0]
    return jnp.linalg.inv(cov).dot(g_ru)


def FD_2E1(F, x_0, h, num_samples, jrandom_key):
    dim = x_0.shape[0]

    forwardX = h*jnp.eye(dim) + x_0
    backwardX = -h*jnp.eye(dim) + x_0

    X = jnp.concatenate([jnp.tile(forwardX, (num_samples, 1)), jnp.tile(backwardX, (num_samples, 1))])
    G = F.f1(X, jrandom_key)
    H = G.T.dot(X - x_0) / (2 * h**2 * num_samples) 

    return H