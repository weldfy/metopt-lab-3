import numpy as np


def uniform(a, b, n):
    return (b - a) * np.random.rand(n, 1) + a


def normal(mu, sigma, n):
    return sigma * np.random.randn((n, 1)) + mu


# returns a * X + b + eps
def linear(X, a, b, eps=1):
    return a * X + b + eps * (np.random.rand(*X.shape) - 0.5)
