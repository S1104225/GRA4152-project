##
#  This module defines a subclass for Poisson GLM
#
import numpy as np
from scipy.stats import poisson
from models.GLMBase import GLMBase

## A subclass that is used to build a Poisson GLM
#
class GLMPoisson(GLMBase):
    ## Constructs the Poisson GLM subclasss
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
        return np.exp(eta)

    ## Evaluates negative loglikelihood for the given model.
    #  @param params matrix with betas
    #  @param x matrix with Xs used for training
    #  @param y matrix with Ys used for training
    #  @return negative loglikelihood for given inputs
    #
    def _neg_llik(self, params, x, y):
        eta = self._get_eta(params, x)
        mu = self._rev_link(eta)
        llik = np.sum((poisson.logpmf(y, mu)))
        return -llik

    ## Prints results of model estimation
    #
    def summary(self):
        self._check_fit()
        print('Poisson GLM has been fit\n================')
        super().summary()