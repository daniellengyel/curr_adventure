import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)

def loss_getter(dim, D, sig, coeff=0.1):
    def helper(X):
        S = X.reshape(dim, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ D @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        
        return 1/4 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S, ord="fro")**4
    return helper

def lmbda_loss(lmbdas, D_diag, sig, coeff, l_max_idx):
    dim = len(D_diag)

    a = D_diag @ lmbdas / len(lmbdas)
    b = jnp.sum(lmbdas)
    return 1/4 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas) + coeff*b**2

def lambda_coeff_zero(D_diag, sig):
    dim = len(D_diag)
    D_diag = jnp.abs(D_diag)
    D_min = jnp.min(D_diag)
    D_sum = jnp.sum(jnp.sqrt(D_diag)) - jnp.sqrt(D_min)
    numerator = 2 * (1 + dim) * D_min * sig**2 + D_sum**2 * sig**2 + jnp.sqrt(D_sum**2 * (8 * (1 + dim) * D_min + D_sum**2) * sig**4)
    
    a = jnp.sqrt(2 * numerator/(dim * D_min))
    l = (2 * (1 + dim) * D_min * sig**2 + numerator)/(D_min**2 * a)

    lmbdas = [(sig * jnp.sqrt(2*l/(D_diag[i] * a))) for i in range(len(D_diag))]
    lmbdas[jnp.argmin(jnp.abs(D_diag))] = l
    
    return jnp.array(lmbdas)

def lambda_coeff_zero_set_max_lambda(D_diag, l_max, sig):
    dim = len(D_diag)
    D_diag = jnp.abs(D_diag)
    D_min = jnp.min(D_diag)
    D_sum = jnp.sum(jnp.sqrt(D_diag)) - jnp.sqrt(D_min)

    sqrt_term = 3 * jnp.sqrt(3 * dim) * jnp.sqrt(27 * dim * D_sum**4 * l_max**2 * sig**4 - 2 * D_min**3 * D_sum**2 * l_max**4 * sig**2)
    cbrt_term = jnp.cbrt(sqrt_term + 27 * dim * D_sum**2 * l_max * sig**2 - D_min**3 * l_max**3)/dim

    a = (dim * cbrt_term + D_min * l_max)**2 / (3 * dim**2 * cbrt_term)
    
    lmbdas = []
    for i in range(len(D_diag)):
        if D_diag[i] != 0:
            lmbdas.append(min(sig * jnp.sqrt(2*l_max/(D_diag[i] * a)), l_max))
        else:
            lmbdas.append(l_max)
        
    lmbdas[jnp.argmin(D_diag)] = l_max
    
    return jnp.array(lmbdas)


def generate_sing_vals_V(D_diag, sig, max_h):
    dim = len(D_diag)
    sig = max(sig, 1e-4) # machine precision roughly 

    if 0 not in D_diag:
        lmbda = lambda_coeff_zero(D_diag, sig)
    else:
        lmbda = lambda_coeff_zero_set_max_lambda(D_diag, max_h**2, sig)
    
    if jnp.max(lmbda) > max_h**2:
        lmbda = lambda_coeff_zero_set_max_lambda(D_diag, max_h**2, sig)

    sing_vals = jnp.diag(lmbda**0.5)

    V = jnp.eye(dim)

    return sing_vals, V



if __name__ == "__main__":
    pass