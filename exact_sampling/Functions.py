import jax.numpy as jnp 
import jax.random as jrandom
from jax import jit, grad, jacfwd

class Quadratic:
    def __init__(self, Q, b, sig=0):
        self.Q = Q
        self.b = b
        self.sig = sig
        self._f1 = grad(lambda x: self.f(x, None)[0])
        self._f2 = jacfwd(lambda x: self.f1(x))
        
        
    def f(self, X, jrandom_key=None):
        is_flat = False
        if len(X.shape) == 1:
            is_flat = True
            X = X.reshape(1, -1)
        Y = jnp.dot(self.Q, X.T)
        Y = jnp.diag(jnp.dot(X, Y)) + X.dot(self.b) # TODO fix. inefficient way to remove x_j^T Q x_i for i != j. 
        if jrandom_key is not None:
            Y += self.sig * jrandom.normal(jrandom_key, shape=(X.shape[0], ))
        if is_flat:
            return Y[0]
        return Y
    
    def f1(self, X):
        return self._f1(X.reshape(1, -1))[0]
    
    def f2(self, X):
        return self._f2(X.reshape(1, -1)).reshape(X.size, X.size)


class Ackley:
    def __init__(self, sig=0):
        if sig is None:
            sig = 0
        self.sig = sig

        self._f1 = grad(lambda x: self.f(x, None)[0])
        self._f2 = jacfwd(lambda x: self.f1(x))
    
    def f(self, X, jrandom_key=None):
        xs = X.T
        out_shape = xs[0].shape
        a = jnp.exp(-0.2 * jnp.sqrt(1. / len(xs) * jnp.square(jnp.linalg.norm(xs, axis=0))))
        b = - jnp.exp(1. / len(xs) *jnp.sum(jnp.cos(2 * jnp.pi * xs), axis=0))
        out = jnp.array(-20 * a + b + 20 + jnp.exp(1)).reshape(out_shape)
        if jrandom_key is not None:
            return out + self.sig * jrandom.normal(jrandom_key, shape=(X.shape[0], )) 
        return out


    def f1(self, X):
        """del H/del xi = -20 * -0.2 * (xi * 1/n) / sqrt(1/n sum_j xj^2) * a + 2 pi sin(2 pi xi)/n * b"""
        # xs = X.T
        # out_shape = xs.shape
        # a = jnp.exp(-0.2 * jnp.sqrt(1. / len(xs) * jnp.square(jnp.linalg.norm(xs, axis=0))))
        # b = -jnp.exp(1. / len(xs) * jnp.sum(jnp.cos(2 * jnp.pi * xs), axis=0))
        # a_p = -0.2 * (xs * 1. / len(xs)) / jnp.sqrt(1. / len(xs) * jnp.square(jnp.linalg.norm(xs, axis=0)))
        # b_p = -2 * jnp.pi * jnp.sin(2 * jnp.pi * xs) / len(xs)
        # grad = jnp.nan_to_num(
        #     -20 * a_p * a + b_p * b).reshape(out_shape)  # only when norm(x) == 0 do we have nan and we know the grad is zero there
        # grad = grad.T
        
        # return grad
        return self._f1(X.reshape(1, -1))[0]

    def f2(self, X):
        # return jacfwd(lambda x: self.f1(x))(X).reshape(X.shape[0], X.shape[1], X.shape[1])
        return self._f2(X.reshape(1, -1)).reshape(X.size, X.size)

class Brown:
    def __init__(self, sig=0):
        self.sig = sig

        self._f1 = grad(lambda x: self.f(x, None)[0])
        self._f2 = jacfwd(lambda x: self.f1(x))
    
    def f(self, X, jrandom_key=None):
        is_flat = False
        if len(X.shape) == 1:
            is_flat = True
            X = X.reshape(1, -1)

        X2 = X**2
        out = jnp.sum(X2[:, :-1]**(X2[:, 1:] + 1) + X2[:, 1:]**(X2[:, :-1] + 1), axis=1)
        if jrandom_key is not None:
            return out + self.sig * jrandom.normal(jrandom_key, shape=(X.shape[0], )) 
        if is_flat:
            return out[0]
        return out
    
    def f1(self, X, jrandom_key=None):
        # X2 = X**2 
        # logX2 = jnp.log(X2)
        # logX2 = jax.ops.index_update(logX2, X2 == 0, 0)
        # grad = jnp.zeros(X.shape)
        # middle_terms = (2*(X2[:, 2:] + 1))*X2[:, 1:-1]**(X2[:, 2:])*X[:, 1:-1] + 2 * X[:, 1:-1] * logX2[:, 2:] * X2[:, 2:]**(X2[:, 1:-1] + 1) \
        #                 + (2*(X2[:, :-2] + 1))*X2[:, 1:-1]**(X2[:, :-2])*X[:, 1:-1] + 2 * X[:, 1:-1] * logX2[:, :-2]  * X2[:, :-2]**(X2[:, 1:-1] + 1)
        # zeroth_term = (2*(X2[:, 1] + 1))*X2[:, 0]**(X2[:, 1])*X[:, 0] + 2 * X[:, 0] * logX2[:, 1] * X2[:, 1]**(X2[:, 0] + 1) 
        # last_term = (2*(X2[:, -2] + 1))*X2[:, -1]**(X2[:, -2])*X[:, -1] + 2 * X[:, -1] * logX2[:, -2]  * X2[:, -2]**(X2[:, -1] + 1)
        
        # grad = jax.ops.index_update(grad, jax.ops.index[:, 1:-1], middle_terms)
        # grad = jax.ops.index_update(grad, jax.ops.index[:, 0], zeroth_term)
        # grad = jax.ops.index_update(grad, jax.ops.index[:, -1], last_term)

        
        # if jrandom_key is not None:
        #     return grad + self.noise_std * jrandom.normal(jrandom_key, X.shape) 
        # return grad 
        return self._f1(X.reshape(1, -1))[0]    

    def f2(self, X):
        # return jacfwd(lambda x: self.f1(x, None))(X).reshape(X.shape[0], X.shape[1], X.shape[1])
        return self._f2(X.reshape(1, -1)).reshape(X.size, X.size)