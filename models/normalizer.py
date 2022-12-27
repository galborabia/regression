import numpy as np


class Normalizer(object):
    def __init__(self):
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        return self.predict(X)

    def predict(self, X):
        # apply normalization
        return (X - self.data_min) / (self.data_max - self.data_min)
