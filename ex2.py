import torch
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit, train_test_split
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

torch.manual_seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)

'''* write a model `Ols` which has a propoery $w$ and 3 methods: `fit`, `predict` and `score`.?
hint: use [numpy.linalg.pinv](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.pinv.html)
to be more efficient. '''


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


# Write a new class OlsGd which solves the problem using gradinet descent.
# The class should get as a parameter the learning rate and number of iteration. 
# Plot the loss convergance. for each alpha, learning rate plot the MSE with respect to number of iterations.
# What is the effect of learning rate? 
# How would you find number of iteration automatically? 
# Note: Gradient Descent does not work well when features are not scaled evenly (why?!).
# Be sure to normalize your feature first.


class Normalizer(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        # apply normalization
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)


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
            X = self.normalizer.predict(X)

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
        X = torch.from_numpy(X.astype(np.float32))
        return torch.mm(X, self.w).flatten()

    def _step(self, X, Y):
        # use w update for gradient descent
        y_pred = self._predict(X)

        loss = torch.square(y_pred - Y).mean()
        loss.backward()

        with torch.no_grad():
            self.w.sub_(self.learning_rate * self.w.grad)
        self.w.grad.zero_()
        return loss.item()


class RidgeLs(Ols):
    def __init__(self, ridge_lambda, *wargs, **kwargs):
        super(RidgeLs, self).__init__(*wargs, **kwargs)
        self.ridge_lambda = ridge_lambda

    def _fit(self, X, Y):
        covariance = np.dot(X.T, X) + np.identity(X.shape[1]) * self.ridge_lambda
        correlation = np.dot(X.T, Y)
        weights = np.dot(np.linalg.pinv(covariance), correlation)
        self.w = weights


# Solution to the questions

boston_X, boston_y = load_boston(return_X_y=True)

number_samples = boston_X.shape[0]
number_features = boston_X.shape[1]

print(f"Number of samples: {number_samples}")  # 506 samples
print(f"Number of features: {number_features}")  # 13 features

# split to train test
X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.25, random_state=42)

# Fit the model. What is the training MSE?
# train / test MSE
ols_model = Ols()
ols_model.fit(X_train, y_train)
train_score = ols_model.score(X_train, y_train)
test_score = ols_model.score(X_test, y_test)
train_predictions = ols_model.predict(X_train)
test_predictions = ols_model.predict(X_test)
print(f"OLS train score {train_score}")
print(f"OLS test score {test_score}")

# Plot a scatter plot where on x-axis plot  Y  and in the y-axis  Y^OLS
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_train, y=train_predictions)
sns.scatterplot(x=y_test, y=test_predictions)
plt.xlabel("true values")
plt.ylabel("predicted values")
plt.title("Boston dataset predicted values vs true values");
plt.legend(["train", "test"]);
plt.show()

# Split the data to 75% train and 25% test 20 times. What is the average MSE now for train and test?
ss = ShuffleSplit(n_splits=20, test_size=.25, random_state=42)
mse_scores_train = []
mse_scores_test = []

for i, (train_index, test_index) in enumerate(ss.split(boston_X)):
    train_x, train_y = boston_X[train_index], boston_y[train_index]
    test_x, test_y = boston_X[test_index], boston_y[test_index]
    ols_model = Ols()
    ols_model.fit(train_x, train_y)
    train_score = ols_model.score(train_x, train_y)
    test_score = ols_model.score(test_x, test_y)
    mse_scores_train.append(train_score)
    mse_scores_test.append(test_score)

print(f"Training mean MSE {np.mean(mse_scores_train)}")
print(f"Test mean MSE {np.mean(mse_scores_test)}")

# Use a t-test to prove that the MSE for training is significantly smaller than for testing. What is the p-value?
stats_train_test, train_test_pvalue = stats.ttest_rel(mse_scores_train, mse_scores_test)

print(f"T-test PValue train vs test results {train_test_pvalue}")
print(f"Statistics train vs test results {stats_train_test}")

'''
Write a new class OrdinaryLinearRegressionGradientDescent which inherits from 
 and solves the problem using gradient descent.
The class should get as a parameter the learning rate and number of iteration. Plot the class convergence.
What is the effect of learning rate?
How would you find number of iteration automatically?
Note: Gradient Descent does not work well when features are not scaled evenly (why?!).
Be sure to normalize your features first.
'''

# before we choose normalization we need to find the distribution of each feature

fig, axs = plt.subplots(1, boston_X.shape[1])
fig.set_size_inches(130, 8)
for feature in range(boston_X.shape[1]):
    plt.figure(1, figsize=(10, 8))
    plt.subplot(1, boston_X.shape[1], feature + 1)
    feature_data = boston_X[:, feature]
    sns.histplot(x=feature_data, stat='density', kde=True, bins=20)
    plt.title(f"Histogram plot for feature X{feature}")
    plt.xlabel(f"X{feature}")
    plt.ylabel("Density")

plt.show()

'''we can see that for feature number 5 (start count from 0), 
it may be better to normalize this features with StandardScaler '''


olsgd = OlsGd(num_iteration=10000, learning_rate=0.01, verbose=False)
olsgd.fit(X_train, y_train)
train_score = olsgd.score(X_train, y_train)
test_score = olsgd.score(X_test, y_test)

print(f"OlsGd train score: {train_score}")
print(f"OlsGd test score: {test_score}")

#  Plot the class convergence.
plt.figure(figsize=(12, 6))
sns.lineplot(y=olsgd.loss_history, x=np.arange(1, len(olsgd.loss_history) + 1))
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("OlsGd error by epoch");
plt.show()

# What is the effect of learning rate?
learning_rates = [0.001, 0.01, 0.05, 0.1]
lr_results = {}
for lr in learning_rates:
    olsgd = OlsGd(num_iteration=1000, learning_rate=lr, verbose=False, early_stop=False)
    olsgd.fit(boston_X, boston_y)
    lr_results[lr] = olsgd.loss_history

lr_results = pd.DataFrame(lr_results)

plt.figure(figsize=(12, 8))

sns.lineplot(data=lr_results)
plt.title("Learning Rate Effect")
plt.ylabel("Mean square error")
plt.xlabel("epoch")
plt.ylim(0, 500)
plt.legend()
plt.show()

# Ridgels
ridgels = RidgeLs(ridge_lambda=1e-2)
ridgels.fit(X_train, y_train)
train_score = ridgels.score(X_train, y_train)
test_score = ridgels.score(X_test, y_test)

print(f"RidgeLs train score: {train_score}")
print(f"RidgeLs test score: {test_score}")

# Use scikitlearn implementation for OLS, Ridge and Lasso
X_train, X_test, y_train, y_test = train_test_split(boston_X, boston_y, test_size=0.25, random_state=42)


def model_evaluation(model, train_x, test_x, train_y, test_y):
    model.fit(train_x, train_y)
    model_train_prediction = model.predict(train_x)
    model_train_mse = mean_squared_error(train_y, model_train_prediction)
    model_test_prediction = model.predict(test_x)
    model_test_mse = mean_squared_error(test_y, model_test_prediction)
    return model_train_mse, model_test_mse


# LinearRegression
linear_regression = LinearRegression()
train_mse, test_mse = model_evaluation(linear_regression, X_train, X_test, y_train, y_test)

print(f"LinearRegression Train MSE: {train_mse}")
print(f"LinearRegression Test MSE: {test_mse}")

# Lasso
lasso = Lasso()
train_mse, test_mse = model_evaluation(lasso, X_train, X_test, y_train, y_test)

print(f"Lasso Train MSE: {train_mse}")
print(f"Lasso Test MSE: {test_mse}")

# Ridge
ridge = Ridge()
train_mse, test_mse = model_evaluation(ridge, X_train, X_test, y_train, y_test)

print(f"Ridge Train MSE: {train_mse}")
print(f"Ridge Test MSE: {test_mse}")
