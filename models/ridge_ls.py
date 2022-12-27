import numpy as np
from models.ols import Ols


class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _fit(self, X, Y):
        X = RidgeLs.pad(X)
        covariance = np.dot(X.T, X) + np.identity(X.shape[1]) * self.ridge_lambda
        correlation = np.dot(X.T, Y)
        weights = np.dot(np.linalg.pinv(covariance), correlation)
        self.w = weights
