from re import X
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
    # F = HeartDisease(sig, noise_type)
    # F_no_noise = HeartDisease(0, noise_type)
    # dim = len(F.opt_X)
    # x_0 = jnp.ones(dim)/jnp.sqrt(dim)
    # return F, x_0
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
    A = jrandom.normal(subkey, shape=(dim, dim,))
    U, _, _ = jnp.linalg.svd(A) # some rotation matrix
    
    Q = U @ jnp.diag(eigs) @ U.T

    jrandom_key, subkey = jrandom.split(jrandom_key)
    b = jrandom.normal(subkey, shape=(dim,))

    F = Quadratic(Q, b, sig, noise_type)

    x_0 = jnp.ones(dim)/jnp.sqrt(dim)
    return F, x_0

def generate_QuadLogPoly(F_name, sig, noise_type):
    dim, space_type, ub, lb, space_type_b, ub_b, lb_b, num_b, seed = F_name.split("_")
    dim, ub, lb, seed = int(dim), float(ub), float(lb), int(seed)
    space_type_b, ub_b, lb_b, num_b = float(space_type_b), float(ub_b), float(lb_b), float(num_b)
    
    jrandom_key = jrandom.PRNGKey(seed)

    if space_type == "log":
        eigs = jnp.logspace(lb, ub, dim)
    else:
        eigs = jnp.linspace(lb, ub, dim)

    jrandom_key, subkey = jrandom.split(jrandom_key)
    A = jrandom.normal(subkey, shape=(dim, dim,))
    U, _, _ = jnp.linalg.svd(A) # some rotation matrix
    Q = 1/2. * U @ jnp.diag(eigs) @ U.T

    jrandom_key, subkey = jrandom.split(jrandom_key)
    Q_b = jrandom.normal(subkey, shape=(dim,))
    Quad_sol = jnp.linalg.solve(2 * Q, Q_b)


    # get barriers
    # This defines a region such that the convex hull of the feasible region 
    # given by an infinite number of barriers forms the level set of U @ jnp.diag(w_Q_eigs) @ U.T 
    # corresponding to 1.
    if space_type == "log":
        w_Q_eigs = jnp.logspace(lb_b, ub_b, dim)
    else:
        w_Q_eigs = jnp.linspace(lb_b, ub_b, dim)

    jrandom_key, subkey = jrandom.split(jrandom_key)
    A = jrandom.normal(subkey, shape=(dim, dim,))
    U, _, _ = jnp.linalg.svd(A) # some rotation matrix
    jrandom_key, subkey = jrandom.split(jrandom_key)
    ws = jrandom.normal(subkey, shape=(dim, num_b))
    ws = ws/jnp.linalg.norm(ws, axis=0)
    ws = U.T @ jnp.diag(w_Q_eigs**1/2.) @ ws 
    bs = jnp.ones(num_b)

    x_0 = jnp.ones(dim)/jnp.sqrt(dim)
    return F, x_0


def load_cutest_quadratic(F_name, lmbda, sig, noise_type):
    """{'HIE1372D': 637, 'KSIP': 20, 'DEGENQPC': 50, 'HS76': 4, 'HS35MOD': 2, 'S268': 5, 
    'AVGASB': 8, 'HS35I': 3, 'DUAL4': 75, 'AVGASA': 8, 'QPCBLEND': 83, 
    'DUAL3': 111, 'DUAL2': 96, 'DUAL1': 85, 'GMNCASE3': 175, 'QPCBOEI1': 384, 'GMNCASE2': 175, 
    'HS76I': 4, 'QPCBOEI2': 143, 'DUALC5': 8, 'DUALC1': 9, 'HS21': 2, 'HS35': 3, 'GMNCASE4': 175, 
    'TABLE7': 624, 'HS51': 5, 'QPCSTAIR': 385, 'TARGUS': 162, 'HS268': 5, 'HS52': 5, 'HS53': 5,"""
    cutest_F = pycutest.import_problem(F_name)
    x0 = cutest_F.x0
    val, grad = cutest_F.obj(x0, True)
    Q = 1/2. * cutest_F.ihess(x0)
    Q_b = grad - 2*Q @ x0
    Q_c = val - Q_b @ x0 - x0.T @ Q @ x0

    const_vals, ws = cutest_F.cons(x0, gradient=True)
    bs = const_vals + ws @ x0

    if lmbda is None:
        lmbda = 1 

    lmbda *= 1/len(ws)

    return QuadLogPolytopeBarrier(Q, Q_b, Q_c, ws, bs, x0, lmbda, sig, noise_type)



class QuadLogPolytopeBarrier:

    def __init__(self, Q, Q_b, Q_c, ws, bs, x0, lmbda, sig=0, noise_type="gaussian"):
        """ws.shape = (N, d), bs.shape = (N)"""
        self.Q = Q
        self.Q_b = Q_b
        self.Q_c = Q_c

        self.ws = jnp.array(ws)
        self.bs = jnp.array(bs)

        self.x0 = x0
        self.dim = len(x0)

        self.lmbda = lmbda
        self.sig = sig
        self.noise_type = noise_type

        self._f1 = grad(lambda x: self.f(x, None))
        self._f2 = jacfwd(lambda x: self.f1(x))
        
    # @partial(jit, static_argnums=(0,))
    def _get_dists(self, X):
        """We consider the sum of log barrier (equivalent to considering each barrier to be a potential function).
        Distance to a hyperplane w.x = b is given by | w.x/|w| - b/|w| |. We consider the absolute value of this, which follows the assumption that if we are on the a side of the hyperplane we stay there. 
        However, the signs tell us whether we are on the side of the hyperplane which is closer to the origin. If the sign is negative, then we are closer."""
        
        X_len_along_ws = X.dot(self.ws.T)/jnp.linalg.norm(self.ws, axis=1)
        hyperplane_dist = self.bs/jnp.linalg.norm(self.ws, axis=1)
        dist = X_len_along_ws - hyperplane_dist # dist.shape = (N_ws)
        signs = 2*(dist * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dist), signs
    
    # @partial(jit, static_argnums=(0,))
    def f(self, X, jrandom_key=None):
        """x.shape = (d). Outside of the bounded region around zero we are infinite."""
        dists, signs = self._get_dists(X) 
        barrier = -jnp.sum(jnp.log(dists))
        barrier = jnp.where(jnp.any(signs > 0), jnp.inf, barrier)
        out = X.T @ self.Q @ X + X @ self.Q_b + self.Q_c + self.lmbda * barrier

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

        print(jnp.linalg.cond(X_data))
        print(jnp.linalg.cond(StandardScaler().fit_transform(X_data)))
        X_std = X_data/100 # StandardScaler().fit_transform(X_data)

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


