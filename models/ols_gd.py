import torch
import numpy as np
from models.ols import Ols
from models.normalizer import Normalizer

# Write a new class OlsGd which solves the problem using gradinet descent.
# The class should get as a parameter the learning rate and number of iteration.
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate?
# How would you find number of iteration automatically?
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!). Be sure to normalize your feature first.
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Normalizer():
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


class OlsGd(Ols):

    def __init__(self, learning_rate=.05,
                 num_iteration=1000,
                 normalize=True,
                 early_stop=True,
                 verbose=True):

        super(OlsGd, self).__init__()
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.early_stop = early_stop
        self.normalize = normalize
        self.normalizer = Normalizer()
        self.verbose = verbose
        self.loss_history = None

    def _fit(self, X, Y, reset=True, track_loss=True):
        # remeber to normalize the data before starting
        if self.normalize:
            X = self.normalizer.fit(X)
        X = OlsGd.pad(X)

        X = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32))
        self.w = torch.randn(X.shape[1], 1, requires_grad=True)
        epochs_loss = []
        for epoch in range(self.num_iteration):

            step_loss = self._step(X, Y)
            epochs_loss.append(step_loss)
            if self.verbose:
                print(f"Epoch {epoch}: loss = {epochs_loss[epoch]}")

            if self.early_stop and epoch >= 2:
                if abs(epochs_loss[epoch - 1] - epochs_loss[epoch]) <= 1e-5:
                    if self.verbose:
                        print(f"Early stopping condition met, number of iteration {epoch}")
                    break

        if track_loss:
            self.loss_history = epochs_loss

    def _predict(self, X):
        if self.normalize:
            X = self.normalizer.predict(X)
        X = OlsGd.pad(X)
        X = torch.from_numpy(X.astype(np.float32))
        return torch.mm(X, self.w).flatten()

    def _step(self, X, Y):
        # use w update for gradient descent
        y_pred = torch.mm(X, self.w).flatten()

        loss = torch.square(y_pred - Y).mean()
        loss.backward()

        with torch.no_grad():
            self.w.sub_(self.learning_rate * self.w.grad)
        self.w.grad.zero_()
        return loss.item()

