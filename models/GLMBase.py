##
#  This module defines a superclass GLMBase for all GLMs
#  
import numpy as np
from scipy.optimize import minimize


## A superclass that defines main methods and properties of all GLMs
#
class GLMBase:
    ## Constructs the GLM superclass
    #  @param x a matrix of exogenous variables with the next format: [p, N]
    #  @param y a matrix of endogenous variables with the next format: [p, N]
    #  @param const a boolean that tells whether the interception point is used
    #
    def __init__(self, x, y, const):
        self._x = x
        self._y = y
        self._params = np.array([])
        self._fit = False
        self._const = const

        self._check_args()

    ## Checks that all inputs are correct
    #
    def _check_args(self):
        assert isinstance(self._x, np.ndarray) and isinstance(self._y, np.ndarray), 'Input data should be numpy arrays'
        assert self._x.shape[1] == self._y.shape[1], 'Matrices have different numbers of observations'
        assert isinstance(self._const, bool), 'const should be boolean'

    ## Checks that the model has been fit
    #
    def _check_fit(self):
        assert self._fit, 'Model has not been fit yet. Please call .fit() first'

    ## Calculates eta by multiplying parameters and endogenous variables
    #  @param params matrix with betas
    #  @param x matrix with Xs
    #  @return matrix multiplication product
    #
    def _get_eta(self, params, x):
        return np.matmul(x.T, params)
    
    ## Defines a reverse model's link function
    #  @param eta matrix multiplication product for betas and Xs
    #  @return matrix of mus calculated by this function
    #
    def _rev_link(self, eta):
        raise NotImplementedError

    ## Evaluates negative loglikelihood for the given model.
    #  @param params matrix with betas
    #  @param x matrix with Xs used for training
    #  @param y matrix with Ys used for training
    #  @return negative loglikelihood for given inputs
    #
    def _neg_llik(self, params, x, y):
        raise NotImplementedError
    
    ## Returns estimated betas to the user
    #
    @property
    def params(self):
        return self._params
    
    ## Evaluates negative loglikelihood for the given model.
    #  @param init_param initial value for betas. Default value is 0.1
    #  @return estimated betas
    #
    def fit(self, init_param = 0.1):
        init_params = np.repeat(init_param, self._x.shape[0])
        results = minimize(self._neg_llik, init_params, args=(self._x, self._y))
        self._params = results['x']
        self._fit = True
        return results['x']
    
    ## Prints results of model estimation
    #
    def summary(self):
        # We add + int(not self._const) since indices start from 0 if the intercept was used and 1 otherwise
        summary = '\n'.join([f'x{i + int(not self._const)}: {value:>12.6f}' for i, value in enumerate(self._params)])
        print(summary)

    ## Estimate values for Ys based on Xs using estimated betas
    #  @param new_x matrix of Xs used to predict Ys
    #  @return matrix with predicated values for Ys
    #
    def predict(self, new_x):
        self._check_fit()
        eta = self._get_eta(self._params, new_x)
        mu = self._rev_link(eta)
        return mu
