
import jax.numpy as jnp
from jax import grad
from scipy.interpolate import RBFInterpolator



class RBF:
    def __init__(self, X, F_vals, smoothing):
        self.rbf = RBFInterpolator(X.T, F_vals, smoothing=smoothing) #, epsilon=0.1, kernel="gaussian")
        self.coeffs = jnp.array(self.rbf._coeffs)
        self.y = jnp.array(self.rbf.y)
        self.epsilon = self.rbf.epsilon
        self.shift = jnp.array(self.rbf._shift)
        self.scale = jnp.array(self.rbf._scale)
        self.powers = jnp.array(self.rbf.powers)
        self.rbf_f1 = grad(lambda x: self._evaluate(x, self.y, self.epsilon, self.coeffs, self.shift, self.scale, self.powers))

    def f(self, X):
        return self._evaluate(X, self.y, self.epsilon, self.coeffs, self.shift, self.scale, self.powers)
        
    def f1(self, X):
        return self.rbf_f1(X)

    def f2(self, X):
        return self._thin_plate_f2(X, self.y, self.epsilon, self.coeffs, self.shift, self.scale, self.powers)
    
    
    def _thin_plate_f2(self, x, y, epsilon, coeffs, shift, scale, powers):
        dim = len(x)
        p = y.shape[0]
        yeps = y*epsilon
        xeps = x*epsilon
        r = jnp.linalg.norm(xeps - yeps, axis=1)

        log_r = jnp.log(r)

        a = 2 * epsilon**2 * jnp.eye(dim) * (log_r @ coeffs[:p, 0])
        b = 2 * epsilon**2 * (xeps - yeps).T @ ((coeffs[:p, 0]/r**2).reshape(-1, 1) * (xeps - yeps))
        c = 2 * epsilon * jnp.eye(dim) * jnp.sum(coeffs[:p, 0])

        return a + b + c
    
    
    def _evaluate(self, x, y, epsilon, coeffs, shift, scale, powers):
        p = y.shape[0]

        yeps = y*epsilon
        xeps = x*epsilon
        xhat = (x - shift)/scale

        r = jnp.linalg.norm(xeps - yeps, axis=1)
        kernel_vec = r**2 * jnp.log(r)
        kernel_vec = jnp.nan_to_num(kernel_vec, 0)
        
        poly_vec = jnp.prod(xhat ** powers, axis=1)

        out = kernel_vec @ coeffs[:p, 0] + poly_vec @ coeffs[p:, 0]

        return out


