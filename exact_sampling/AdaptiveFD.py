import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jax.config import config
config.update("jax_enable_x64", True)


class adapt_FD:
    def __init__(self, sig, rl=1.5, ru=6):
        self.sig = sig
        self.rl = rl 
        self.ru = ru 

    def grad(self, f, x_0, jrandom_key, H=None):
        dim = len(x_0)

        all_dirs = jnp.eye(dim)
        grad = np.zeros(dim)
        total_func_calls = 0
        jrandom_key, subkey = jrandom.split(jrandom_key)
        f_x_0 = f.f(x_0, subkey)
        total_func_calls += 1
        for i in range(dim):
            curr_dir = all_dirs[i]
            jrandom_key, subkey = jrandom.split(jrandom_key)
            grad[i], curr_num_func_calls = self._adapt_FD(f, x_0, curr_dir, self.sig, self.rl, self.ru, subkey, f_x_0)
            total_func_calls += curr_num_func_calls
        return grad, total_func_calls
    

    def _adapt_FD(self, f, x_0, x_dir, sig, rl, ru, jrandom_key, f_x_0=None):
        h = 2./jnp.sqrt(3) * jnp.sqrt(sig)
        l = 0
        u = None

        total_func_calls = 0
        total_func_calls_limit = 250
        if f_x_0 is None:
            jrandom_key, subkey = jrandom.split(jrandom_key)
            f_x_0 = f.f(x_0, subkey)
            total_func_calls += 1

        jrandom_key, subkey = jrandom.split(jrandom_key)
        f_x_0_four_h = f.f(x_0 + 4 * h * x_dir, subkey)
        jrandom_key, subkey = jrandom.split(jrandom_key)
        f_x_0_h = f.f(x_0 + h * x_dir, subkey)
        
        total_func_calls += 2
        
        while total_func_calls < total_func_calls_limit:
            curr_r = jnp.abs(f_x_0_four_h - 4 * f_x_0_h + 3 * f_x_0)/(8*sig)

            if curr_r < rl:
                l = h
            elif curr_r > ru:
                u = h 
            else:
                break

            if u is None:
                h = 4 * h
                f_x_0_h = f_x_0_h
                jrandom_key, subkey = jrandom.split(jrandom_key)
                f_x_0_four_h = f.f(x_0 + 4 * h * x_dir, subkey)
                total_func_calls += 1
            elif l == 0:
                h = h/4.
                f_x_0_four_h = f_x_0_h
                jrandom_key, subkey = jrandom.split(jrandom_key)
                f_x_0_h = f.f(x_0 + h * x_dir, subkey)
                total_func_calls += 1
            else:
                h = (l + u)/2.
                jrandom_key, subkey = jrandom.split(jrandom_key)
                f_x_0_four_h = f.f(x_0 + 4 * h * x_dir, subkey)
                jrandom_key, subkey = jrandom.split(jrandom_key)
                f_x_0_h = f.f(x_0 + h * x_dir, subkey)
                total_func_calls += 2

        return (f_x_0_h - f_x_0)/h, total_func_calls

