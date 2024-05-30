import numpy as np

from sgd import SGD
from utils import linear_transform


def sgd():
    def grad(X_batch, y_batch, w, learning_rate, epoch, **kwargs):
        error = (np.dot(X_batch, w) - y_batch) / len(X_batch)
        return learning_rate * 2 * np.dot(X_batch.T, error)

    return SGD(grad, linear_transform)
