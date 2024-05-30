import numpy as np

from utils import min_max_scaler_std, min_max_scaler_scaled, mse


class SGD:
    def __init__(self, grad=None, data_transform=None):
        self.info = {'epochs': None,
                     'batch_size': None,
                     'learning_rate': None,
                     'results': [],
                     'mse': []}
        self.w = None
        self.X_info = {'data': None, 'min': None, 'max': None}
        self.data_transform = data_transform
        if self.data_transform is None:
            self.data_transform = lambda x: x
        self.y_info = {'data': None, 'min': None, 'max': None}
        self.grad = grad

    def train(self, X, y, batch_size, learning_rate, epochs, calc_metrics, **kwargs):
        self.info['epochs'] = epochs
        self.info['batch_size'] = batch_size
        self.info['learning_rate'] = learning_rate
        X = np.array(X, dtype="float64")
        y = np.array(y, dtype="float64")
        X, y = self.scale_and_save(X, y)
        X = self.data_transform(X)
        self.fit(X, y, batch_size, learning_rate, epochs, calc_metrics, **kwargs)


    def predict(self, X):
        if self.w is None:
            return None
        return linear_predict(X, self.w, self.data_transform, self.scale, self.rescale)

    def scale_and_save(self, X, y):
        self.__save(X, self.X_info)
        self.__save(y, self.y_info)
        return self.scale(X), self.scale(y)

    def scale(self, data):
        return min_max_scaler_std(data)

    def rescale(self, data):
        return min_max_scaler_scaled(data, self.y_info['min'], self.y_info['max'])

    def __save(self, data, info):
        info['data'] = data
        info['min'] = data.min(axis=0)
        info['max'] = data.max(axis=0)

    def fit(self, X, y, batch_size, learning_rate, epochs, calc_metrics, **kwargs):
        self.info['results'] = []
        self.info['mse'] = []
        w = np.zeros((X.shape[1], 1), dtype='float64')
        n = len(y)
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, n, batch_size):
                begin = i
                end = min(n, i + batch_size)
                X_batch = X_shuffled[begin: end]
                y_batch = y_shuffled[begin: end]
                w -= self.grad(X_batch, y_batch, w, learning_rate, epoch, **kwargs)
            if calc_metrics:
                self.info['results'].append(w)
                self.info['mse'].append(mse(y_shuffled, X_shuffled @ w))
        self.w = w

def linear_predict(X, w, transform, scale, rescale):
    X = scale(X)
    X = transform(X)
    return rescale(X @ w)