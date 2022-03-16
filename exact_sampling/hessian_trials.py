import jax.numpy as jnp
import jax.random as jrandom
from matplotlib import use
import matplotlib.pyplot as plt

from optimize_USG_error import optimize_uncentered_S
from Functions import Quadratic, Ackley, Brown

from tqdm import tqdm

def simplex_gradient(F, x_0, S, subkey_f):
    jrandom_key, subkey = jrandom.split(subkey_f)
    FS = F.f(S.T + x_0, subkey)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = F.f(x_0.reshape(1, -1), subkey)
    SS_inv = jnp.linalg.inv(S.dot(S.T))
    return SS_inv.dot(S.dot(FS - F_x_0))

def simplex_gradient_vals(x_0, S, F_vals):
    pass

def simplex_hessian(F, x_0, S, subkey_f):
    jrandom_key, subkey = jrandom.split(subkey_f)
    FS = F.f(S.T + x_0, subkey)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    F_x_0 = F.f(x_0.reshape(1, -1), subkey)
    SS_inv = jnp.linalg.inv(S.dot(S.T))

    F_grad = SS_inv.dot(S.dot(FS - F_x_0))
    print(F_grad)
    print(F.f1(x_0))
    print()
    grads = []

    S_prime = jnp.concatenate([jnp.zeros(shape=(len(x_0), 1)), S], axis=1) + x_0.reshape(-1, 1)
    FS_prime = jnp.concatenate([F_x_0, FS]) 

    print(S_prime)


    for i in range(S_prime.shape[1]):
        curr_S = jnp.concatenate([S_prime[:, :i], S_prime[:, i+1:]], axis=1) - S_prime[:, i].reshape(-1, 1)
        curr_F = jnp.concatenate([FS_prime[:i], FS_prime[i+1:]]) - FS_prime[i]
        curr_SS_inv = jnp.linalg.inv(curr_S.dot(curr_S.T))
        grads.append(curr_SS_inv.dot(curr_S.dot(curr_F)))      

        print(curr_S)  
    
        print(curr_SS_inv.dot(curr_S.dot(curr_F)))
        print(F.f1(S_prime[:, i]))
        print()




seed = 0
dim = 4
sig = 0.0001
F = Ackley(sig)

jrandom_key = jrandom.PRNGKey(seed)

x_0 = jnp.array(list(range(1, dim+1)))
x_0 /= jnp.linalg.norm(x_0)


num_opt_steps = 15
H = F.f2(x_0)
S, opt_loss = optimize_uncentered_S(H, sig, max_steps=num_opt_steps)

jrandom_key, subkey = jrandom.split(jrandom_key)

# simplex_hessian(F, x_0, S, subkey)

# ++++ Quadratic ++++
jrandom_key_quad = jrandom.PRNGKey(0)
jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)

# jrandom_key_quad, subkey = jrandom.split(jrandom_key_quad)
# b = jrandom.normal(subkey, shape=(dim,))
# F = Quadratic(Q, b, 0)

dim = 5
N = dim * 1000
Q = jrandom.normal(subkey, shape=(dim, dim,))
Q = jnp.diag(jnp.linspace(0.5, 100, dim))
S = jrandom.normal(subkey, shape=(dim, N, ))


print(jnp.mean(jnp.array([s.T.dot(Q.dot(s)) * jnp.outer(s, s) for s in S.T]), axis=0))
print(jnp.cov(S).dot(Q).dot(jnp.cov(S)))


