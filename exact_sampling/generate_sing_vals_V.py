import jax.numpy as jnp
from jax import grad
import jax.random as jrandom
import numpy as np

import scipy 

from jax.config import config
config.update("jax_enable_x64", True)



def loss_getter(dim, D, sig, coeff=0.1):
    def helper(X):
        S = X.reshape(dim, dim).T
        
        S_inv = jnp.linalg.inv(S)
        
        first_term = S_inv.T @ jnp.diag(S.T @ D @ S)
        second_term = jnp.linalg.norm(S_inv, ord="fro")**2
        third_term = S_inv.T @ jnp.ones(dim)
        third_term = jnp.linalg.norm(third_term)**2
        
        return 1/2 * jnp.linalg.norm(first_term)**2 + sig**2 * (second_term + third_term) + coeff*jnp.linalg.norm(S, ord="fro")**4
    return helper

def get_alpha(D, l1, l2):
    Dmax = jnp.max(D)
    Dmin = jnp.min(D)
    return -1/(Dmax - Dmin) * (Dmin + l2/(l1 - l2) * jnp.sum(D)) 

def get_V_star(dim, alpha):
    I = jnp.eye(dim)
    v1 = jnp.sqrt(I[0] + alpha * (I[-1] - I[0]))
    
    V_tmp = jnp.concatenate([v1.reshape(1, -1), I[1:]]).T
    V, _ = jnp.linalg.qr(V_tmp)
    return V

def get_lambda_star(dim, sig, coeff):
    
    if dim == 3:
        l1, l2 = jnp.cbrt(sig**2 / coeff), jnp.cbrt(sig**2 / (2*coeff))
    else:
        l1_denom = coeff*sig**2
        
        l1_factor = jnp.sqrt(coeff**2 * (dim + 1) * sig**4)
        
        
        cbrt_term = jnp.cbrt(coeff * sig**2 * (dim - 1 - jnp.sqrt(dim + 1)) / (2 * coeff**2 * dim * (dim - 3)) )
        
        l1 = l1_factor / l1_denom * cbrt_term
        l2 = cbrt_term
    lambda_star = jnp.concatenate([jnp.array([l1]), l2 * jnp.ones(dim - 1)])
    return lambda_star

def lmbda_loss(lmbdas, D_diag, sig, coeff, lmbda_star):
    dim = len(D_diag)

    a = D_diag @ lmbda_star / len(lmbda_star)
    if a < 0:
        l_max_idx = jnp.argmax(D_diag)
    else:
        l_max_idx = jnp.argmin(D_diag)


    a = D_diag @ lmbdas / len(lmbdas)
    b = jnp.sum(lmbdas)
    return 1/2 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas) + coeff*b**2

    
def get_lambda_tilde(D_diag, sig, coeff, lmbda_star, eps_bound=1e-5):
    dim = len(D_diag)
    lmbdas_init = np.ones(dim)
    bounds = tuple((eps_bound, None) for _ in range(dim))

    a = D_diag @ lmbda_star / len(lmbda_star)
    if a < 0:
        l_max_idx = jnp.argmax(D_diag)
    else:
        l_max_idx = jnp.argmin(D_diag)

    def lmbda_loss(lmbdas):
        a = D_diag @ lmbdas / len(lmbdas)
        b = jnp.sum(lmbdas)
        return 1/2 * a**2 * dim / lmbdas[l_max_idx] + sig**2 * dim/lmbdas[l_max_idx] + sig**2 * jnp.sum(1/lmbdas) + coeff*b**2

    lmbda_tilde = jnp.array(scipy.optimize.minimize(lmbda_loss, lmbdas_init, bounds=bounds)["x"])

    return lmbda_tilde

def get_multi_lambda_tilde(D_diag_multi, sig, coeff, lmbda_star_multi, eps_bound=1e-8):

    a_multi = [D_diag_multi[i] @ lmbda_star_multi[i] / len(lmbda_star_multi[i]) for i in range(len(D_diag_multi))]
    l_max_idx_multi = [jnp.argmax(D_diag_multi[i]) if a_multi[i] < 0 else jnp.argmin(D_diag_multi[i]) for i in range(len(D_diag_multi))]
    total_dim = sum([len(D_diag_multi[i]) for i in range(len(D_diag_multi))])

    dims = jnp.array([len(D_diag_multi[i]) for i in range(len(D_diag_multi))])

    
    run_sum = 0
    l_max_idx_multi_all = []
    dims_all = []
    a_filter = np.zeros(shape=(len(D_diag_multi), total_dim))
    for i in range(len(dims)):
        l_max_idx_multi_all += dims[i] * [l_max_idx_multi[i] + run_sum]
        dims_all += dims[i] * [dims[i]]
        a_filter[i, run_sum:run_sum + dims[i]] = 1
        run_sum += dims[i]

    a_filter = jnp.array(a_filter)
    
    l_max_idx_multi_all = tuple([l_max_idx_multi_all])
    dims_all = jnp.array(dims_all)
    D_multi = jnp.concatenate(D_diag_multi)

    def lmbda_loss(lmbdas):

        a = a_filter @ ((D_multi / (jnp.sqrt(dims_all) * jnp.sqrt(lmbdas[l_max_idx_multi_all])) * lmbdas))
        a_squared = a.T @ a
        b = a_filter @ lmbdas
        b_squared = b.T @ b


        # for i in range(len(D_diag_multi)):
            # dim = len(D_diag_multi[i])
            # a = D_diag_multi[i] @ lmbdas[curr_lmbda_num:curr_lmbda_num+dim] / dim
            # b = jnp.sum(lmbdas[curr_lmbda_num:curr_lmbda_num+dim])
            # curr_loss += 1/2 * a**2 * dim / lmbdas[curr_lmbda_num + l_max_idx_multi[i]] + sig**2 * dim/lmbdas[curr_lmbda_num + l_max_idx_multi[i]] + sig**2 * jnp.sum(1/lmbdas[curr_lmbda_num:curr_lmbda_num+dim]) + coeff*b**2
            # curr_lmbda_num += dim
        loss = 1/2 * a_squared + sig**2 * jnp.sum(1/lmbdas[l_max_idx_multi_all]) + sig**2 * jnp.sum(1/lmbdas) + coeff*b_squared
        return loss
    
    tmp_lmbda_tilde = jnp.array([0.0350607 , 0.01449902, 0.01412839, 0.0138914, 0.3246133 , 0.03795886, 0.03197942, 0.03197662, 0.03193183,
             0.03190124, 0.03177178, 0.03175972, 0.03175547, 0.03174929,
             0.03172634, 0.03168808, 0.03168544, 0.03166968, 0.03166923,
             0.03166431, 0.03166135, 0.03159596, 0.03141259, 0.03133268,
             0.03009546, 0.0283492 , 0.02817361, 0.02799253, 0.02747393,
             0.02703282, 0.02694328, 0.02677607, 0.02657705, 0.02652984,
             0.02645184, 0.02640589, 1.08824979, 0.052379  , 0.04977457, 0.04721447, 0.04717982,
             0.04713195, 0.04713451, 0.04709396, 0.04701821, 0.04706659,
             0.04698764, 0.04694285, 0.04691369, 0.04689094, 0.0468909 ,
             0.04688808, 0.04688388, 0.04687826, 0.04687509, 0.04687344,
             0.04686102, 0.0468593 , 0.0468465 , 0.04684606, 0.04684285,
             0.04683243, 0.04683158, 0.04681068, 0.04680965, 0.04680523,
             0.04679303, 0.04678836, 0.04675685, 0.04661435, 0.04651847,
             0.04613409, 0.04596316, 0.04577467, 0.04556416, 0.04541263,
             0.04510022, 0.0451219 , 0.04467399, 0.0441646 , 0.04393059,
             0.04346917, 0.04324267, 0.04272223, 0.04250405, 0.0422871 ,
             0.04203621, 0.04149565, 0.04115429, 0.04087367, 0.04076232,
             0.04037356, 0.0401241 , 0.04005248, 0.03955809, 0.03946576,
             0.03929919, 0.03909493, 0.03901273, 0.01616137])**2

    print(lmbda_loss(tmp_lmbda_tilde))

    lmbdas_init = np.ones(total_dim)
    bounds = tuple((eps_bound, None) for _ in range(total_dim))
    
    lmbda_tilde = jnp.array(scipy.optimize.minimize(lmbda_loss, lmbdas_init, bounds=bounds,  jac=grad(lmbda_loss))["x"])
    print(lmbda_loss(lmbda_tilde))
    lambda_tilde_multi = []
    curr_lmbda_num = 0
    for i in range(len(D_diag_multi)):
        dim = len(D_diag_multi[i])
        lambda_tilde_multi.append(lmbda_tilde[curr_lmbda_num:curr_lmbda_num+dim])
        curr_lmbda_num += dim


    return lambda_tilde_multi


def permute_rows(M, i, j):
    tmp_row = M[i].copy()
    M[i] = M[j].copy()
    M[j] = tmp_row
    return M


# # This function may change if we decide to allow for lambda_star solutions. 
def generate_sing_vals_V(D_diag, sig, coeff):
    dim = len(D_diag) 

    lambda_star = get_lambda_star(dim, sig, coeff)
    lmbda = get_lambda_tilde(D_diag, sig, coeff, lambda_star, eps_bound=1e-15)
    sing_vals = jnp.diag(lmbda**0.5)
    V = jnp.eye(dim)

    return sing_vals, V


# This function may change if we decide to allow for lambda_star solutions. 
def generate_sing_vals_V_multi(D_diag_multi, sig, coeff):


    lambda_star_multi = [get_lambda_star(len(D_diag_multi[i]), sig, coeff) for i in range(len(D_diag_multi))]
    lmbda_multi = get_multi_lambda_tilde(D_diag_multi, sig, coeff, lambda_star_multi, eps_bound=1e-8)
    sing_vals = [lmbda_multi[i]**0.5 for i in range(len(D_diag_multi))]
    V_multi = [jnp.eye(len(D_diag_multi[i])) for i in range(len(D_diag_multi))]

    return sing_vals, V_multi


def get_lambda_tilde_least_squares(D_diag, sig, coeff, lmbda_star, eps_bound=1e-5):
    dim = len(D_diag)
    lmbdas_init = np.ones(dim)
    bounds = tuple((eps_bound, None) for _ in range(dim))

    a = D_diag @ lmbda_star / len(lmbda_star)
    if a < 0:
        l_max_idx = jnp.argmax(D_diag)
    else:
        l_max_idx = jnp.argmin(D_diag)

    def helper(x):
        a, b, l = x[0], x[1], x[2]
        
        D_seq = 1/jnp.sqrt(D_diag * a / l + b)

        f1 = 1/dim * (sig * (jnp.sum((D_diag * D_seq)[:l_max_idx]) + jnp.sum((D_diag * D_seq)[l_max_idx+1:])) + D_diag[l_max_idx] * l) - a
        f2 = 2 * (sig * (jnp.sum(D_seq[:l_max_idx]) + jnp.sum(D_seq[l_max_idx+1:])) + l) - b
        f3 = a * D_diag[l_max_idx] * l - 1/2. * a**2 * dim - sig**2 *(dim + 1) + b * l**2
        
        return jnp.array([f1, f2, f3])**2
    
    
    res = optimize.least_squares(helper, [a, 10., 1.], method="dogbox", tr_solver='exact', jac=lambda x: jacfwd(helper)(jnp.array(x)), bounds=([-np.inf, 0., 0.], np.inf))
    print(res)
    if res["cost"] > 1:
        res = get_lambda_tilde(D_diag, sig, coeff, lmbda_star, eps_bound=1e-5)
    
    a, b, l = res["x"]
    
    lmbda_tilde = [float(jnp.sqrt(1/(a * D_diag[i] / l + b))) for i in range(l_max_idx)] + [l] + [float(jnp.sqrt(1/(a * D_diag[i] / l + b))) for i in range(l_max_idx+1, dim)]

    return jnp.array(lmbda_tilde)


if __name__ == "__main__":
    D_diag = jnp.array([3, 4, 5, 6])
    lmbda = get_lambda_tilde(D_diag, 1, 1, get_lambda_star(len(D_diag), 1, 1), eps_bound=1e-6)
    print(lmbda)
    a = D_diag @ lmbda / len(lmbda)
    print(a)

    # D_diag_multi = [jnp.array([1, 200, -3, -4]), jnp.array([5, 6, 7, 8, 9, 10, 11, 12])]

    # D_diag_multi =  [jnp.array([675.20051706, 813.58855034, 902.38153221, 965.54902346]), jnp.array([ 232.42521254,  334.54360068,  664.07578131,  664.30886894,
    #           668.05512491,  670.62525508,  681.61012856,  682.64592253,
    #           683.01139697,  683.54053652,  685.5230432 ,  688.83433237,
    #           689.06439288,  690.43799398,  690.47708125,  690.90722949,
    #           691.16594061,  696.90956342,  713.32621891,  720.61622704,
    #           846.6184271 , 1075.33577212, 1102.38031622, 1131.16493491,
    #          1219.00519823, 1300.57189438, 1317.92713172, 1351.1817099 ,
    #          1392.20586847, 1402.0444181 , 1418.56433088, 1428.62262654]), jnp.array([   96.97920809,   445.58329169,   541.87230141,
    #            664.71465559,   665.29227901,   666.0511765 ,
    #            666.96999075,   669.27220119,   672.07173395,
    #            673.61245399,   676.83277458,   678.4606098 ,
    #            680.0739581 ,   682.15884262,   682.16484015,
    #            682.66599669,   683.45128368,   684.21563395,
    #            684.5582067 ,   684.72214789,   685.81472461,
    #            685.95674572,   686.99779281,   687.03279669,
    #            687.29354153,   688.12046891,   688.18386053,
    #            689.68421996,   689.75528463,   690.05726011,
    #            690.86515663,   691.16436287,   693.06274524,
    #            701.5216982 ,   707.02710026,   728.73396968,
    #            737.93426139,   747.96890143,   759.16418986,
    #            771.16489945,   784.39052623,   798.35031424,
    #            829.43925305,   864.22609599,   883.22288421,
    #            923.02826319,   943.4464293 ,   986.80858122,
    #           1010.19563407,  1031.57409601,  1056.28423607,
    #           1103.60944914,  1149.74978764,  1174.81884908,
    #           1195.41949823,  1239.24306126,  1261.28777128,
    #           1280.35580794,  1335.98079325,  1366.74715305,
    #           1379.42123945,  1411.80442234,  1425.11697458,
    #          48323.8052095 ])]

    # sig = 0.1
    # coeff = 0.1
    # print(generate_sing_vals_V_multi(D_diag_multi, sig, coeff)[0])
    # print(repr(jnp.diag(generate_sing_vals_V(D_diag_multi[0], sig, coeff)[0])))
    # print(repr(jnp.diag(generate_sing_vals_V(D_diag_multi[1], sig, coeff)[0])))
    # print(repr(jnp.diag(generate_sing_vals_V(D_diag_multi[2], sig, coeff)[0])))

