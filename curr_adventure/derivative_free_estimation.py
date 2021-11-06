from jax import random as jrandom
import jax.numpy as jnp
import jax
from scipy.optimize import fsolve
from jax import jit, partial

# We introduce the following notation:
# iEj := ith order Estimator using jth order information with i <= j. 
# If i=j then the estimator is exact. 


def multilevel_beta_inv_2E1(F, x_0, alpha, N, d_prime, jrandom_key):
    d = len(x_0)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey) 
    ru = samples_points - jnp.mean(xs, axis=0)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    out_grads = F.f1(sample_points, subkey)
    grad_X = grads.T.dot(ru)/float(N) # new_proper_cov(sample_points, out_grads)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    grad_X_multilevel = U.dot(jnp.linalg.inv(U.T.dot(grad_X).dot(U))).dot(U.T)
    return np_new_cov(sample_points).dot(grad_X_multilevel)

def multilevel_beta_newton_update_2E1_1E1(F, x_0, alpha, N, d_prime, jrandom_key):
    """Makes use of Matrix-Vector Product."""
    d = len(x_0)

    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T # (d, d')

    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey, chosen_basis_idx=U_idxs)  
    ru = (sample_points - jnp.mean(sample_points, axis=0)).T # (d, N)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    out_grads = F.f1(sample_points, subkey)

    gradF = F.f1(jnp.array([x_0]))[0]
    grad_X_low_inv = jnp.linalg.inv(out_grads.T[U_idxs].dot(ru[U_idxs].T)/float(N))
    cov = beta_covariance(x_0.shape[0], R, alpha) # jnp.cov(sample_points.T)
    return -cov*U.dot(grad_X_low_inv.dot(U.T.dot(gradF)))



def get_A(xs):
    """xs.shape = [N, d]"""
    xs -= jnp.mean(xs, axis=0)
    curr_m = (xs ** 2).T.dot(xs ** 2)
    curr_m = curr_m / len(xs)
    return curr_m

def beta_second_shift_estimator(F, x_0, alpha, N, jrandom_key):
    jrandom_key, subkey = jrandom.split(jrandom_key)
    sample_points, radius = jax_hit_run(x_0, F, x_0.shape[0], N, alpha, subkey)
    sample_points = jnp.concatenate([sample_points, 2*x_0 - sample_points])
    out_points = F.f(sample_points)
    diffs = sample_points -  jnp.mean(sample_points, axis=0) 
    second_shift_est =  diffs.T.dot(diffs * out_points.reshape(-1, 1)) / len(diffs)

    second_shift_est -= F.f(x_0.reshape(1, -1))[0] * diffs.T.dot(diffs)/len(diffs) 

    A = get_A(sample_points) 
    
    return  2*((second_shift_est - jnp.eye(len(x_0)) * jnp.diag(second_shift_est))/(A*2.) + (jnp.eye(len(x_0)) * jnp.linalg.inv(A).dot(jnp.diag(second_shift_est))))

def neystrom_inv(H, d_prime, jrandom_key):
    d = len(H)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    A_inv_approx = U.dot(jnp.linalg.inv(U.T.dot(H).dot(U))).dot(U.T)
    return A_inv_approx

def neystrom_update_direction(H, d_prime, grad, jrandom_key):
    """Makes use of matrix-vector product."""
    d = len(H)
    jrandom_key, subkey = jrandom.split(jrandom_key)
    U_idxs = jrandom.choice(subkey, a=d, shape=(d_prime,), replace=False)
    U = jnp.eye(d)[U_idxs].T
    dir_approx = U.dot(jnp.linalg.inv(U.T.dot(H).dot(U)).dot(U.T.dot(grad)))
    return dir_approx

# Quadratic Regression
def get_quadratic_data_matrix(xs):
    """xs.shape = (N, d). out.shape = (N, d(d + 1)/2)"""
    one_two_factors = np.array([1] + [2]*(xs.shape[1])) # I should have one less two, but since i cant do slicing with [:-0] for the loop to get X, i need to add one here so i can start at [:-1]
    out = [xs[:, i].reshape((-1, 1)) * xs[:, i:] * one_two_factors[:-(i+ 1)]  for i in range(xs.shape[1])] 
    return np.hstack(out)

def quadratic_regression(xs, y):
    X = get_quadratic_data_matrix(xs)
    return np.linalg.inv(np.dot(np.transpose(X), X)).dot(X.T.dot(y))

def quadratic_regression_ipm(F, x_0, alpha, N, control_variate=True, estimated_gradient=False):
    sample_points = hit_run(x_0, F, x_0.shape[0], N, alpha)    
    Y = 2*(F.f(sample_points) - F.f(np.array([x_0]))[0] - (sample_points - x_0).dot(F.f1(np.array([x_0])).reshape(-1)))
    return quadratic_regression(sample_points, Y)


# BFGS approximation
    
def BFGS_update(F, x_0, x_1, inv_hessian_approx=None):
    if inv_hessian_approx is None:
        H = np.eye(len(x_0))
    else:
        H = inv_hessian_approx

    grad_diff = (F.f1(np.array([x_1])) - F.f1(np.array([x_0])))[0]
    update_step = x_1 - x_0
    
    ys = np.inner(grad_diff, update_step)
    Hy = np.dot(H, grad_diff)
    yHy = np.inner(grad_diff, Hy)
    H += (ys + yHy) * np.outer(update_step, update_step) / ys ** 2
    H -= (np.outer(Hy, update_step) + np.outer(update_step, Hy)) / ys
    return H    


def solve_for_h(err_bound, R, theta=1):
    c = err_bound
    a = 2 * (theta + 2 * theta**0.5)**6
    second_numerator = ((2/3.)**(1/3.) * a)
    second_denominator = (-9 * a * c**2 * R + 3**0.5 *(4 * a**3 * c**3 + 27 * a**2 * c**4 * R**2)**0.5)**(1/3.)
    
    third_numerator = (-9 * a * c**2 * R + 3**0.5 * (4 * a**3 * c**3 + 27 * a**2 * c**4 * R**2)**0.5)**(1/3.)
    third_denominator = (2**(1/3.) * 3**(2/3.) * c)
    return R - second_numerator/second_denominator + third_numerator/third_denominator