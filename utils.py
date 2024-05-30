import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme()


def describe(X, y):
    df = pd.DataFrame({'X': X, 'y': y})
    return sns.scatterplot(data=df, x='X', y='y')


def mse(y, y_pred):
    return ((y - y_pred) ** 2).mean(axis=0).item()


def min_max_scaler_std(data):
    min_value = data.min(axis=0)
    max_value = data.max(axis=0)
    return (data - min_value) / (max_value - min_value)


def min_max_scaler_scaled(data_std, min_value, max_value):
    return data_std * (max_value - min_value) + min_value


def polynom_transform(data, degree):
    power_arrays = []
    for power in range(degree + 1):
        power_arrays.append(data ** power)
    return np.hstack(power_arrays)


def linear_transform(data):
    return polynom_transform(data, 1)
