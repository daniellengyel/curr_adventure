from sqlite3 import adapt
import jax.numpy as jnp 
import jax.random as jrandom
from jax import jit, grad, jacfwd
import pycutest

from sklearn.preprocessing import StandardScaler 

import pandas as pd 

import os 
HOME = os.getenv("HOME") 

def get_F(F_type, F_name, sig, noise_type):
    
    return generate_quadratic(F_name, sig, noise_type)

def generate_quadratic(F_name, sig, noise_type):
    dim, space_type, ub, lb, seed = F_name.split("_")
    dim, ub, lb, seed = int(dim), float(ub), float(lb), int(seed)

    jrandom_key = jrandom.PRNGKey(seed)

    if space_type == "log":
        eigs = jnp.logspace(lb, ub, dim)
    else:
        eigs = jnp.linspace(lb, ub, dim)

    jrandom_key, subkey = jrandom.split(jrandom_key)
    Q = jrandom.normal(subkey, shape=(dim, dim,))
    Q = 1/2. * Q @ Q.T 
    Q = Q + jnp.diag(eigs)

    jrandom_key, subkey = jrandom.split(jrandom_key)
    b = jrandom.normal(subkey, shape=(dim,))

    F = Quadratic(Q, b, sig, noise_type)

    x_0 = jnp.ones(dim)/jnp.sqrt(dim)
    return F, x_0




class Quadratic:
    def __init__(self, Q, b, sig=0,  noise_type="gaussian"):
        self.Q = Q
        self.b = b
        self.sig = sig
        self.noise_type = noise_type
        self._f1 = grad(lambda x: self.f(x, None))
        self._f2 = jacfwd(lambda x: self.f1(x))
        
        
    def f(self, X, jrandom_key=None):
        out = X.T @ self.Q @ X + X.dot(self.b)
        
        if jrandom_key is not None:
            if self.noise_type == "uniform":
                eps = self.sig * jnp.sqrt(3)
                return out + 2 * eps * jrandom.uniform(jrandom_key) - eps
            else:
                return out + self.sig * jrandom.normal(jrandom_key) 
    
        return out
    
    def f1(self, X):
        return self._f1(X)
    
    def f2(self, X):
        return self._f2(X).reshape(X.size, X.size)


class PyCutestWrapper:
    def __init__(self, cutest_f, sig=0,  noise_type="gaussian"):
        self.cutest_f = cutest_f
        self.sig = sig
        self.noise_type = noise_type

    def f(self, X, jrandom_key=None):
        out = self.cutest_f.obj(X)
        if jrandom_key is not None:
            if self.noise_type == "uniform":
                eps = self.sig * jnp.sqrt(3)
                return out + 2 * eps * jrandom.uniform(jrandom_key) - eps
            else:
                return out + self.sig * jrandom.normal(jrandom_key) 
        return out

    def f1(self, X):
        return self.cutest_f.obj(X, True)[1]

    def f2(self, X):
        return self.cutest_f.hess(X)


def PyCutestGetter(func_name=None, func_dim=None, func_i=None, dim_i=None, sig=0, noise_type="gaussian"):

    adapt_functions = {
        "AIRCRFTB": [5], "ALLINITU": [4], "ARWHEAD": [100], 
        "BARD": [3], "BDQRTIC":[100], "BIGGS3":[3], "BIGGS5":[5], "BIGGS6":[6], 
        "BOX2":[2], "BOX3":[3], "BRKMCC":[2], "BROWNAL":[10, 100, 200], "BROWNDEN":[4], 
        "CLIFF":[2], "CRAGGLVY":["C", [4, 10, 50, 100], ['M', 1, 4, 24, 49]], "CUBE":[2], 
        "DENSCHNA":[2], "DENSCHNB":[2], "DENSCHNC":[2], "DENSCHND":[3], "DENSCHNE":[3], "DENSCHNF":[2], 
        "DIXMAANA": ["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANB":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANC":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAAND":["C", [15, 90, 300], ['M', 5, 30, 100]], 
        "DIXMAANE":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANF":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANG":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANH":["C", [15, 90, 300], ['M', 5, 30, 100]], 
        "DIXMAANI":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANJ":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANK":["C", [15, 90, 300], ['M', 5, 30, 100]], "DIXMAANL":["C", [15, 90, 300], ['M', 5, 30, 100]],
        "DQRTIC":[10, 50, 100], 
        "EDENSCH":[36], "EIGENALS":["C", [6, 110], ['N', 2, 10]], "EIGENBLS":["C", [6, 110], ['N', 2, 10]], "EIGENCLS":["C", [30], ['M', 2]], "ENGVAL1":[2, 50, 100], "EXPFIT":[2], 
        "FLETCBV3":[10, 100], "FLETCHBV":[10, 100], "FREUROTH":[2, 10, 50, 100], 
        "GENROSE":[5, 10, 100], "GULF":[3], 
        "HAIRY":[2], "HELIX":[3], 
        "JENSMP": [2], 
        "KOWOSB":[4], 
        "MEXHAT":[2], "MOREBV": [10, 50, 100], 
        "NCB20B":[21, 22, 50, 100, 180], "NONDIA": [10, 20, 30, 50, 90, 100], "NONDQUAR": [100], 
        "OSBORNEA":[5], "OSBORNEB":[11], 
        "PENALTY1":[4, 10, 50, 100], "PFIT1LS":[3], "PFIT2LS":[3], "PFIT3LS":[3], "PFIT4LS":[3], 
        "QUARTC":[25, 100], 
        "SINEVAL":[2], "SINQUAD":[5, 50, 100], "SISSER":[2], "SPARSQUR":[10, 50, 100], 
        "TOINTGSS":[10, 50, 100], "TQUARTIC":[5, 10, 50, 100], "TRIDIA":[10, 20, 30, 50, 100], 
        "WATSON":[12, 31], "WOODS": ["C", [4, 100], ['NS', 1, 25]], 
        "ZANGWIL2":[2]
    }


    try:
        if func_name is None:
            func_name = list(adapt_functions.keys())[func_i]
        
        if func_dim is None:
            if adapt_functions[func_name][0] == "C":
                func_dim = adapt_functions[func_name][1][dim_i]
            else:
                func_dim = adapt_functions[func_name][dim_i]

        if adapt_functions[func_name][0] == "C":
            dim_i = adapt_functions[func_name][1].index(func_dim)
            sif_key, sif_val = adapt_functions[func_name][2][0], adapt_functions[func_name][2][dim_i + 1]
        else:
            sif_key, sif_val = 'N', func_dim

    except:
        # print("Function index {} with dimension index {} is not a test problem.".format(func_i, dim_i))
        return None, None, None
    
    

    try:
        curr_prob = pycutest.import_problem(func_name)
    except:
        print("Could not import problem {}.".format(func_name))
        return func_name, None, None

    if curr_prob.n != func_dim:
        try:
            curr_prob = pycutest.import_problem(func_name, sifParams={sif_key:sif_val})
        except:
            print("Failed with Problem {}, it has {} dimensions instead of the expected {} dimensions.".format(func_name, curr_prob.n, func_dim))
            return func_name, None, None

    return  func_name, curr_prob.x0, PyCutestWrapper(curr_prob, sig, noise_type)


class HeartDisease:
    """"https://www.analyticsvidhya.com/blog/2022/03/logistic-regression-on-uci-dataset/"""

    def __init__(self, sig=0,  noise_type="gaussian"):
        self.sig = sig
        self.noise_type = noise_type

        self.X_data = None
        self.y_data = None
        self._init_data()

        self._f1 = grad(lambda x: self.f(x, None))
        self._f2 = jacfwd(lambda x: self.f1(x))

        self.opt_X = jnp.array([-0.00549042, -0.83499842,  0.90540353, -0.35528798,
                                -0.28677555,  0.06359361,  0.23585258,  0.59582554,
                                -0.49040073, -0.70567094,  0.33217319, -0.85092055,
                                -0.22942091])

    def _init_data(self):
        df = pd.read_csv(HOME + "/curr_adventure/exact_sampling/Datasets/" + 'heart_disease_dataset_UCI.csv')
        X_data = jnp.array(df.iloc[:,0:13].values) 
        y_data = jnp.array(df.iloc[:,13].values)

        X_std = StandardScaler().fit_transform(X_data)

        self.X_data = X_std
        self.y_data = y_data

    def f(self, X, jrandom_key=None):
        lin_model = self.X_data @ X[:13] + X[-1]
        out = 1 / (1 + jnp.exp(-lin_model))

        out = -(1 / len(self.y_data)) * jnp.sum(self.y_data * jnp.log(out) + (1 - self.y_data) * jnp.log(1 - out))

        if jrandom_key is not None:
            if self.noise_type == "uniform":
                eps = self.sig * jnp.sqrt(3)
                return out + 2 * eps * jrandom.uniform(jrandom_key) - eps
            else:
                return out + self.sig * jrandom.normal(jrandom_key) 
    
        return out

    def f1(self, X):
        return self._f1(X)
    
    def f2(self, X):
        return self._f2(X).reshape(X.size, X.size)

