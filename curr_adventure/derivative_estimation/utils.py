from jax import random as jrandom
import jax.numpy as jnp
from .one_E_zero import SD_1E0

class OneEZero():
    def __init__(self, config):
        self.config = config

    def f1(self, F, X, config, jrandom_key, M=None):
        grid_config = {"F": F, "N": config["optimization_meta"]["N"], "h": config["optimization_meta"]["h"], "is_uniform_sphere_random": False}
        if self.config["grad_estimate_type"] == "FD_uniform":
            grid_config["ellipse_M"] = None
            return SD_1E0(F, X[0], grid_config, jrandom_key, sample_based=True)
        elif self.config["grad_estimate_type"] == "FD_Hess":
            grid_config["ellipse_M"] = M
            return SD_1E0(F, X[0], grid_config, jrandom_key, sample_based=True)
        elif self.config["grad_estimate_type"] == "FD_adaptive":
            grid_config["ellipse_M"] = None
            return SD_1E0(F, X[0], grid_config, jrandom_key, sample_based=True)
        else:
            return F.f1(X)[0]
