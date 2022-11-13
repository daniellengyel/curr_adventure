import jax.numpy as jnp
import numpy as np
from jax import jit

from jax.config import config
config.update("jax_enable_x64", True)


def lmbda_loss(lmbdas, D_diag, sig, l_max_idx):
    dim = len(D_diag)

    a = D_diag @ lmbdas / len(lmbdas)
    return 1/4 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas)

@jit
def lambda_no_max_h(D_diag, sig):
    dim = len(D_diag)
    D_diag = jnp.abs(D_diag)
    D_min = jnp.min(D_diag)
    D_sum = jnp.sum(jnp.sqrt(D_diag)) - jnp.sqrt(D_min)
    numerator = 2 * (1 + dim) * D_min * sig**2 + D_sum**2 * sig**2 + jnp.sqrt(D_sum**2 * (8 * (1 + dim) * D_min + D_sum**2) * sig**4)
    
    a = jnp.sqrt(2 * numerator/(dim * D_min))
    l = (2 * (1 + dim) * D_min * sig**2 + numerator)/(D_min**2 * a)

    lmbdas = sig * jnp.sqrt(2*l/(D_diag * a))
    lmbdas = lmbdas.at[jnp.argmin(jnp.abs(D_diag))].set(l)
    
    return jnp.array(lmbdas)


def lambda_coeff_zero_set_max_lambda(D_diag, sig, l_max):
    dim = len(D_diag)

    if dim == 1:
        return jnp.array([min([l_max, 4 * sig / jnp.abs(D_diag)[0]])])

    if sum(D_diag) < 0:
        D_diag = -D_diag

    if all(D_diag > 0):
        lmbda_no_max = lambda_no_max_h(D_diag, sig)
        if all(lmbda_no_max <= l_max):
            return lmbda_no_max

    D_arg_sorted = jnp.argsort(D_diag)

    n_upper = len(D_diag)
    n_lower = sum(D_diag <= 0)
    n_curr = (n_upper + n_lower)//2
    
    def lmbda_helper(num_h):
        K1 = jnp.sum(jnp.sqrt(D_diag[D_arg_sorted[num_h:]]))
        K2 = jnp.sum(D_diag[D_arg_sorted[:num_h]])
        
        c1 = sig * jnp.sqrt(dim * l_max / 2) * K1
        c2 = l_max * K2
        
        disc = 81 * c1**2 - 12 * c2**3 
        if disc < 0:
            r = jnp.sqrt(12 * c2**3)
            theta = jnp.arccos(9 * c1 / r)
            mu1 = jnp.cos(theta/3)
            a = mu1**2 * c2 * 4/3.
            
        else:
            
            cbrt = jnp.cbrt(jnp.sqrt(disc) + 9 * c1)
            a_sqrt = jnp.cbrt(2/3) * c2 / cbrt + cbrt/(jnp.cbrt(2 * 9))
        
            a = a_sqrt**2


        lmbda = np.ones(dim) * float(l_max) 
        lmbda[np.array(D_arg_sorted[num_h:])] = sig * np.sqrt(dim * l_max / (2 * a * D_diag[D_arg_sorted[num_h:]]))

        return lmbda
    
    while True:

        lmbda = lmbda_helper(n_curr)
        
        if n_upper == n_lower:
            break
        
        if any(lmbda > l_max):
            if (n_curr == n_lower) and (n_lower + 1 == n_upper):
                n_lower = n_upper
            else:
                n_lower = n_curr
        else:
            n_upper = n_curr

        n_curr = (n_upper + n_lower) // 2

    return jnp.array(lmbda)
    


def generate_sing_vals_V(D_diag, sig, max_h):
    dim = len(D_diag)
    sig = max(sig, 1e-14) # machine precision roughly

    lmbda = lambda_coeff_zero_set_max_lambda(D_diag, sig, max_h**2)
    # print(lmbda)
    # print(D_diag)
    sing_vals = jnp.diag(lmbda**0.5)
    V = jnp.eye(dim)

    return sing_vals, V



if __name__ == "__main__":
    pass