import numpy as np

import sgd
from utils import polynom_transform


def sgd_l1(degree, l1=0.1):
    def regular_func(w1):
        return l1 * np.sign(w1)

    return sgd_regular(regular_func, degree)


def sgd_l2(degree, l2=0.1):
    def regular_func(w1):
        return l2 * 2 * w1

    return sgd_regular(regular_func, degree)


def sgd_elastic(degree, l1=0.1, l2=0.1):
    def regular_func(w1):
        return l2 * 2 * w1 + l1 * np.sign(w1)

    return sgd_regular(regular_func, degree)


def sgd_regular(regular_func, degree):
    def grad(X_batch, y_batch, w, learning_rate, epoch, **kwargs):
        error = (np.dot(X_batch, w) - y_batch) / len(X_batch)
        return learning_rate * (2 * np.dot(X_batch.T, error) + regular_func(w))

    def data_transform(data):
        return polynom_transform(data, degree)

    return sgd.SGD(grad, data_transform)


