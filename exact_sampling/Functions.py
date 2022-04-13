from sqlite3 import adapt
import jax.numpy as jnp 
import jax.random as jrandom
from jax import jit, grad, jacfwd
import pycutest

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
        return self._f1(X.reshape(1, -1))[0]

    def f2(self, X):
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
    
    def f1(self, X):
        return self._f1(X.reshape(1, -1))[0]    

    def f2(self, X):
        return self._f2(X.reshape(1, -1)).reshape(X.size, X.size)


class PyCutestWrapper:
    def __init__(self, cutest_f, eps=0,  noise_type="gaussian"):
        self.cutest_f = cutest_f
        self.eps = eps
        self.noise_type = noise_type

    def f(self, X, jrandom_key=None):
        out = self.cutest_f.obj(X)
        if jrandom_key is not None:
            if self.noise_type == "uniform":
                return out + 2 * self.eps * jrandom.uniform(jrandom_key) - self.eps
            else:
                return out + self.eps * jrandom.normal(jrandom_key) 
        return out

    def f1(self, X):
        return self.cutest_f.obj(X, True)[1]

    def f2(self, X):
        return self.cutest_f.hess(X)


def PyCutestGetter(i, eps=0, noise_type="gaussian"):
    adapt_functions = ["AIRCRFTB", "ALLINITU", "ARWHEAD", "BARD", "BDQRTIC", "BIGGS3", "BIGGS5", "BIGGS6", "BOX2", "BOX3", "BRKMCC", "BROWNAL", "BROWNDEN", "CLIFF", "CRAGGLVY", "CUBE", "DENSCHND", "DENSCHNE", "DIXMAANH", "DQRTIC", "EDENSCH", "EIGENALS", "EIGENBLS", "EIGENCLS", "ENGVAL1", "EXPFIT", "FLETCBV3", "FLETCHBV", "FREUROTH", "GENROSE", "GULF", "HAIRY", "HELIX", "NCB20B", "NONDIA", "NONDQUAR", "OSBORNEA", "OSBORNEB", "PENALTY1", "PFIT1LS", "PFIT2LS", "PFIT3LS", "PFIT4LS", "QUARTC", "SINEVAL", "SINQUAD", "SISSER", "SPARSQUR", "TOINTGSS", "TQUARTIC", "TRIDIA", "WATSON", "WOODS", "ZANGWIL"]
    adapt_function_dims = [5, 4, 100, 3, 100, 3, 5, 6, 2, 3, 2, 100, 4, 2, 
                            100, 2, 3, 3, 90, 100, 36, 110, 110, 30, 100, 2, 100, 100, 
                            100, 100, 3, 2, 3, 100, 100, 100, 5, 11, 100, 3, 3, 3,
                            3, 100, 2, 100, 2, 100, 100, 100, 100, 31, 100, 2]



    try:
        curr_prob = pycutest.import_problem(adapt_functions[i])
    except:
        print("Could not import problem {}.".format(adapt_functions[i]))
        return adapt_functions[i], None, None

    if curr_prob.n != adapt_function_dims[i]:
        try:
            curr_prob = pycutest.import_problem(adapt_functions[i], sifParams={'N':adapt_function_dims[i]})
        except:
            print("Failed with Problem {}, it has {} dimensions instead of the expected {} dimensions.".format(adapt_functions[i], curr_prob.n, adapt_function_dims[i]))
            return adapt_functions[i], None, None
    return adapt_functions[i], curr_prob.x0, PyCutestWrapper(curr_prob, eps, noise_type)




    # if idx_iterator is None:
    #     idx_iterator = range(len(adapt_functions))
    # for i in idx_iterator:
    #     try:
    #         curr_prob = pycutest.import_problem(adapt_functions[i])
    #     except:
    #         print("Could not import problem {}.".format(adapt_functions[i]))
    #         continue
    #     if curr_prob.n != adapt_function_dims[i]:
    #         try:
    #             curr_prob = pycutest.import_problem(adapt_functions[i], sifParams={'N':adapt_function_dims[i]})
    #         except:
    #             print("Failed with Problem {}, it has {} dimensions instead of the expected {} dimensions.".format(adapt_functions[i], curr_prob.n, adapt_function_dims[i]))
    #             continue
    #     yield adapt_functions[i], curr_prob.x0, PyCutestWrapper(curr_prob, eps, noise_type)


