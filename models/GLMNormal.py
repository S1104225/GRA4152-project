##
#  This module defines a subclass for Normal GLM
#
import numpy as np
from scipy.stats import norm
from models.GLMBase import GLMBase

## A subclass that is used to build a Normal GLM
#
class GLMNormal(GLMBase):
    ## Constructs the Normal GLM subclass
    #  @param x a matrix of exogenous variables with the next format: [p, N]
    #  @param y a matrix of endogenous variables with the next format: [p, N]
    #  @param const a boolean that tells whether the interception point is used
    #
    def __init__(self, x, y, const):
        super().__init__(x, y, const)
    
    ## Defines a reverse model's link function
    #  @param eta matrix multiplication product for betas and Xs
    #  @return matrix of mus calculated by this function
    #
    def _rev_link(self, eta):
        return eta

    ## Evaluates negative loglikelihood for the given model.
    #  @param params matrix with betas
    #  @param x matrix with Xs used for training
    #  @param y matrix with Ys used for training
    #  @return negative loglikelihood for given inputs
    #
    def _neg_llik(self, params, x, y):
        eta = self._get_eta(params, x)
        mu = self._rev_link(eta)
        llik = np.sum((norm.logpdf(y, mu)))
        return -llik

    ## Prints results of model estimation
    #
    def summary(self):
        self._check_fit()
        print(f'Normal GLM has been fit\n================')
        super().summary()