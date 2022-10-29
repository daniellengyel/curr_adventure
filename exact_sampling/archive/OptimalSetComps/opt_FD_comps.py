import jax.numpy as jnp

def get_sampling_set_loss(H, sig, coeff, jrandom_key, loss):
    D_diag, U_H = jnp.linalg.eigh(H)
    S = jnp.diag(2 * jnp.sqrt(sig / jnp.abs(D_diag)))
    return [loss(U_H.dot(S), jrandom_key)]

