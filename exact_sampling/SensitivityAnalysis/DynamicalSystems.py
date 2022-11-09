import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, grad, jacfwd, jacrev
from jax.lax import fori_loop
from jax import jit
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)



class Laser:
    def __init__(self, x_init, y_init, z_init, gamma, dt, final_T, sig=0, noise_type="gaussian", output_var="x"):
        self.x_init = x_init 
        self.y_init = y_init
        self.z_init = z_init
        self.gamma = gamma # this should be fixed 
        
        self.sig = sig
        self.noise_type = noise_type
        self.output_var = output_var
        
        self.final_T = final_T
        self.dt = dt
        
        self.store = {}
        
    def f(self, X, jrandom_key=None):
        X = tuple([float(c) for c in X])
        if X in self.store:
            out_all = self.store[X]
        else:
            out_all = self._fast_f(X)
            self.store[X] = out_all

        if self.output_var == "x":
            out = out_all[0]
        elif self.output_var == "y":
            out = out_all[1]
        else:
            out = out_all[2]

        if jrandom_key is not None:
            if self.noise_type == "uniform":
                eps = self.sig * jnp.sqrt(3)
                return out + 2 * eps * jrandom.uniform(jrandom_key) - eps
            else:
                return out + self.sig * jrandom.normal(jrandom_key) 
        return out
    
    
    @partial(jit, static_argnums=(0,))
    def _f(self, X):
        
        epsilon, delta, alpha, sigma = X
        gamma = self.gamma

        x, y, z = self.x_init, self.y_init, self.z_init

        dt = self.dt
        N = int(self.final_T/dt + 0.5)
        

        def body_fun(i, val):
            x, y, z = val
            
            x_delta = x * (y - 1)
            y_delta = gamma * (delta - y + alpha * (x + z)/(1 + sigma*(x + z)) - x*y)
            z_delta = -epsilon * (z + x)

            x = x + dt*x_delta
            y = y + dt*y_delta
            z = z + dt*z_delta
            return [x, y, z]

        out = [x, y, z]
        out = fori_loop(1, N, body_fun, out)
        return out
    
    def get_full_path(self, X):
        epsilon, delta, alpha, sigma = X
        gamma = self.gamma

        x, y, z = self.x_init, self.y_init, self.z_init

        dt = self.dt
        N = int(self.final_T/dt + 0.5)
        
        res = [[x, y, z]]
        
        for _ in range(N):
            
            x_delta = x * (y - 1)
            y_delta = gamma * (delta - y + alpha * (x + z)/(1 + sigma*(x + z)) - x*y)
            z_delta = -epsilon * (z + x)

            x = x + dt*x_delta
            y = y + dt*y_delta
            z = z + dt*z_delta
            res.append([x, y, z])

        return res


class LaserFull:
    def __init__(self, S_init, N_init, I_init, dt, final_T, normalize_const_diag=jnp.ones(11), sig=0, noise_type="gaussian", output_var="S"):
        self.S_init = S_init 
        self.N_init = N_init
        self.I_init = I_init
        
        self.sig = sig
        self.noise_type = noise_type
        self.output_var = output_var
        
        self.final_T = final_T
        self.dt = dt

        self.normalize_const_diag = normalize_const_diag
        
        self.store = {}
        
    def f(self, X, jrandom_key=None):
        X = X @ jnp.linalg.inv(self.normalize_const_diag)
        X = tuple([float(c) for c in X])
        if X in self.store:
            out_all = self.store[X]
        else:
            out_all = self._f(X)
            self.store[X] = out_all

        if self.output_var == "S":
            out = out_all[0]
            sig_use = self.sig * self.S_init
        elif self.output_var == "N":
            out = out_all[1]
            sig_use = self.sig * self.N_init
        else:
            out = out_all[2]
            sig_use = self.sig * self.I_init

        if jrandom_key is not None:
            if self.noise_type == "uniform":
                eps = sig_use * jnp.sqrt(3)
                return out + 2 * eps * jrandom.uniform(jrandom_key) - eps
            else:
                return out + sig_use * jrandom.normal(jrandom_key) 
        return out
    
    
    @partial(jit, static_argnums=(0,))
    def _f(self, X):
        gamma_0, gamma_c, gamma_f, I_0, A, s_prime, e, V, k, g, N_t = X

        dt = self.dt
        num_steps = int(self.final_T/dt + 0.5)
        
        S, N, I = self.S_init, self.N_init, self.I_init

        def body_fun(i, val):
            S, N, I = val
            
            S_delta = (g * (N - N_t) - gamma_0) * S
            N_delta = (I_0 + A*I/(1 + s_prime * I))/(e*V) - gamma_c * N - g*(N - N_t)*S
            I_delta = - gamma_f * I + k*(g*(N - N_t) - gamma_0)*S

            S = S + dt*S_delta
            N = N + dt*N_delta
            I = I + dt*I_delta
            return [S, N, I]

        out = [S, N, I]
        out = fori_loop(1, num_steps, body_fun, out)
        return out

    
    def get_full_path(self, X):
        gamma_0, gamma_c, gamma_f, I_0, A, s_prime, e, V, k, g, N_t = X

        dt = self.dt
        num_steps = int(self.final_T/dt + 0.5)
        
        S, N, I = self.S_init, self.N_init, self.I_init

        res = [[S, N, I]]
        
        for _ in range(num_steps):
            
            S_delta = (g * (N - N_t) - gamma_0) * S
            N_delta = (I_0 + A*I/(1 + s_prime * I))/(e*V) - gamma_c * N - g*(N - N_t)*S
            I_delta = -gamma_f * I + k*(g*(N - N_t) - gamma_0)*S

            S = S + gamma_0*dt*S_delta
            N = N + gamma_0*dt*N_delta
            I = I + gamma_0*dt*I_delta
            res.append([S, N, I])

        return res

def invert_Laser_params(sigma, alpha, delta, gamma, epsilon, x_init, y_init, z_init):
    gamma_f = epsilon 
    
    gamma_0 = 1
    gamma_c = gamma / gamma_0
    
    s_prime = 5 
    k = 1
    g = s_prime * k * gamma_c / sigma
    
    A = 1
    e = 2.7 
    V = A / (e * gamma_0 * alpha) * k
    
    I_0 = 1
    N_t = (1/gamma_c)*(I_0/(e * V) - delta*(gamma_c * gamma_0)/g)
    
    S_init = x_init * gamma_c / g
    N_init = y_init * gamma_0 / g + N_t
    I_init = (k * gamma_c) / g * (z_init + g * S_init / gamma_c)
    
    return gamma_0, gamma_c, gamma_f, I_0, A, s_prime, e, V, k, g, N_t, S_init, N_init, I_init
    
    



class Monod():
    def __init__(self, S_init, X_init, dt, final_T, sig=0, noise_type="gaussian", output_var="S"):
        self.S_init = S_init
        self.X_init = X_init
        self.dt = dt
        self.final_T = final_T
        self.sig = sig
        self.noise_type = noise_type
        
        self.store = {}
        self.output_var = output_var
    
    
    def f(self, X, jrandom_key=None):
        X = tuple([float(c) for c in X])
        if X in self.store:
            SX = self.store[X]
        else:
            SX = self._f(X)
            self.store[X] = SX

        if self.output_var == "S":
            out = SX[0]
        else:
            out = SX[1]

        if jrandom_key is not None:
            if self.noise_type == "uniform":
                eps = self.sig * jnp.sqrt(3)
                return out + 2 * eps * jrandom.uniform(jrandom_key) - eps
            else:
                return out + self.sig * jrandom.normal(jrandom_key) 
        return out
    
    @partial(jit, static_argnums=(0,))
    def _f(self, X):
        
        mu_max, Y, K_S, K_d = X
#         K_S = K_S * 0.1

        S = self.S_init
        X = self.X_init

        dt = self.dt
        N = int(self.final_T/dt + 0.5)

        def body_fun(i, val):
            S, X = val
            
            S_delta = -1/Y * (mu_max * S)/(K_S + S)*X
            X_delta = (mu_max * S)/(K_S + S) * X - K_d * X

            S = S + dt*S_delta
            X = X + dt*X_delta
            return [S, X]

        SX = [S, X]
        SX = fori_loop(1, N, body_fun, SX)
        return SX



    def get_full_path(self, X):
        mu_max, Y, K_S, K_d = X
        S, X = self.S_init, self.X_init

        dt = self.dt
        N = int(self.final_T/dt + 0.5)
        
        res = [[S, X]]
        
        for _ in range(N):
            
            S_delta = -1/Y * (mu_max * S)/(K_S + S)*X
            X_delta = (mu_max * S)/(K_S + S) * X - K_d * X

            S = S + dt*S_delta
            X = X + dt*X_delta
            res.append([S, X])

        return res
    
    
    