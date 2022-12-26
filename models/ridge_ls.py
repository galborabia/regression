import torch
from models.olg_gd import OlsGd


class RidgeLs(OlsGd):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _step(self, X, Y):
        # use w update for gradient descent
        y_pred = self._predict(X)

        # using regularization on the loss function
        loss = torch.square(y_pred - Y).mean() + self.ridge_lambda * torch.square(self.w).sum()
        loss.backward()

        with torch.no_grad():
            self.w.sub_(self.learning_rate * self.w.grad)
        self.w.grad.zero_()
        return loss.item()
