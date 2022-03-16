import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import multiprocessing

# We expect X to be a (N, d) array, where d is the dimensionality and N is the number of datapoints. 
# The output is then then (N) dimensional. We are only working with scalar functions. 

# output of f1 is of shape (N, d)

# output of f2 is of shape (N, d, d)

# if self.automatic_diff:
#     H_inv = jnp.linalg.inv(jax.hessian(self.combined_F.f)(X).reshape(X.shape[1], X.shape[1]))
#     f1 = jax.grad(lambda x: self.combined_F.f(x)[0])(X)[0]

def get_obj(config, jrandom_key=None):
    # get potential
    if config["Optimization_Problem"]["name"] == "Linear":
        # Set Linear Objective
        if config["Optimization_Problem"]["obj_direction"] == "ones":
            c = jnp.ones(config["dim"])
        else:
            raise ValueError("Does not support given function {} with direction {}.".format(config["potential_name"], potential_meta["direction_name"]))
        F = Linear(c)
    else:
        raise ValueError("Does not support given function {}".format(config["potential_name"]))
    return F  

class Quadratic:
    def __init__(self, Q, b):
        self.Q = Q
        self.Q_inv = jnp.linalg.inv(Q)
        self.b = b
        
    def f(self, X):
        Y = np.dot(self.Q, X.T)
        Y = jnp.diag(jnp.dot(X, Y)) + X.dot(self.b) # TODO fix. inefficient way to remove x_j^T Q x_i for i != j. 
        return Y
    
    def f1(self, X):
        Y = 2*jnp.dot(self.Q, X.T)
        return Y.T + self.b
    
    def f2(self, X):
        return 2 * np.array([list(self.Q)] * X.shape[0])

    def f2_inv(self, X):
        return 1/2. * np.array([list(self.Q_inv)] * X.shape[0])
    

class LinearCombination():

    def __init__(self, obj, barrier, weights):
        self.obj = obj
        self.barrier = barrier
        self.funcs = [self.obj, self.barrier]
        self.weights = weights

    def f(self, X, jrandom_key=None):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f(X, jrandom_key)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f1(self, X):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f1(X)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f2(self, X):
        res = jnp.sum(jnp.array([jnp.multiply(w, f.f2(X)) for w, f in zip(self.weights, self.funcs)]), axis=0)
        return res

    def f2_inv(self, X):
        pre_inv = jnp.array(self.f2(X))
        return jnp.array([jnp.linalg.inv(pre_inv[i]) for i in range(len(pre_inv))])

    def dir_dists(self, xs, dirs):
        return self.barrier.dir_dists(xs, dirs)

class Linear:
    def __init__(self, c):
        """c.shape = (d)"""
        self.c = c

    def f(self, X, jrandom_key=None):
        return X.dot(self.c) #/ float(len(self.c))

    def f1(self, X, jrandom_key=None):
        return np.tile(self.c, (X.shape[0], 1)) #/ float(len(self.c))

    def f2(self, X):
        return np.tile(np.array([0]), (X.shape[0], X.shape[1], X.shape[1]))  #/ float(len(self.c))
