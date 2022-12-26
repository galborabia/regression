import torch
import numpy as np
from sklearn.metrics import mean_squared_error


class Ols(object):
    def __init__(self):
        self.w = None

    @staticmethod
    def pad(X):
        # add column with constant value of 1 for bias
        return np.pad(X, ((0, 0), (1, 0)), constant_values=1)

    def fit(self, X, Y):
        X = Ols.pad(X)
        return self._fit(X, Y)

    def _fit(self, X, Y):
        covariance = np.dot(X.T, X)
        correlation = np.dot(X.T, Y)
        weights = np.dot(np.linalg.pinv(covariance), correlation)
        self.w = weights

    def predict(self, X):
        # return wx
        X = Ols.pad(X)
        return self._predict(X)

    def _predict(self, X):
        # optional to use this
        return np.dot(X, self.w)

    def score(self, X, Y):
        predicted_labels = self.predict(X)
        if isinstance(predicted_labels, torch.Tensor):
            predicted_labels = predicted_labels.detach().numpy()

        return mean_squared_error(Y, predicted_labels)
