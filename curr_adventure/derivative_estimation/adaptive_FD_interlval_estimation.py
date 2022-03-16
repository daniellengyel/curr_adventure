import jax.numpy as np



def adaptive_interval(testing_ratio, r_l, r_u, h_0, eta, jrandom_key):
    h = h_0
    l = 0
    u = float("inf")
    while True:
        jrandom_key, subkey = jrandom.split(jrandom_key)
        tr_curr = testing_ratio(h, subkey)
        if tr_curr < r_l:
            l = h
        elif tr_curr > r_u:
            u = h
        else:
            break
        if u == float("inf"):
            h = eta * h
        elif l == 0:
            h = h / eta
        else:
            h = (l + u)/2.

    return h

def CD_testing_ratio(F, x, p, eps_f):
    p = p / jnp.linalg.norm(p)
    def helper(h, jrandom_key):
        numer = F.f((x + 3*h*p).reshape(1, -1), jrandom_key) - 3 * F.f((x + h*p).reshape(1, -1), jrandom_key) + 3 * F.f((x - h * p).reshape(1, -1), jrandom_key) - F.f((x - 3 * h * p).reshape(1, -1), jrandom_key)
        return abs(numer[0]) / (8 * eps_f)

    return helper