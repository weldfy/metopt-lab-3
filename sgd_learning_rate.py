import math

import numpy as np
from math import pow, exp

import sgd
from utils import linear_transform


def sgd_step_decay(drop=0.5, epochs_drop=10):
    def step_decay(epoch, learning_rate, **kwargs):
        return learning_rate * pow(drop, math.floor((1 + epoch) / epochs_drop))

    return sgd.SGD(gradient_learning_rate(step_decay))


def sgd_exp_decay(k=0.1):
    def exp_decay(epoch, learning_rate, **kwargs):
        return learning_rate * exp(-k * epoch)

    return sgd.SGD(gradient_learning_rate(exp_decay), linear_transform)


def gradient_learning_rate(learning_rate_func):
    def grad(X_batch, y_batch, w, learning_rate, epoch, **kwargs):
        error = (np.dot(X_batch, w) - y_batch) / len(X_batch)
        return 2 * learning_rate_func(epoch, learning_rate, **kwargs) * np.dot(X_batch.T, error)
    return grad
