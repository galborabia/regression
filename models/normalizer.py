from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Normalizer(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        # apply normalization
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
