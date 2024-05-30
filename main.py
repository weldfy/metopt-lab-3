from sklearn.model_selection import train_test_split

import data
import sgd
import sgd_batch
import numpy as np

import sgd_learning_rate
import sgd_regular
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sgd_other
import test

np.random.seed(42)
ATTRS = [
    {'model': sgd_batch.sgd(), 'name': 'batch', 'params': {}},
    {'model': sgd_learning_rate.sgd_exp_decay(k=0.1), 'name': 'exp_decay(0.1)', 'params': {}},
    {'model': sgd_learning_rate.sgd_step_decay(drop=0.5, epochs_drop=10), 'name': 'step_decay(0.5, 10)', 'params': {}},
    {'model': sgd_other.SGDKeras(model_name='Nesterov'), 'name': 'keras.nesterov', 'params': {'momentum': 0.1}},
    {'model': sgd_other.SGDKeras(model_name='Adagrad'), 'name': 'keras.adagrad', 'params': {'initial_accumulator_value': 0.2}},
    {'model': sgd_other.SGDKeras(model_name='RMSprop'), 'name': 'keras.rmrprop',
     'params': {'rho': 0.8, 'momentum': 0.1}},
    {'model': sgd_other.SGDKeras(model_name='SGD'), 'name': 'keras.momentum',
     'params': {'momentum': 0.1}},
    {'model': sgd_other.SGDKeras(model_name='Adam'), 'name': 'keras.momentum',
     'params': {'beta_1': 0.9, 'beta_2': 0.99}},

]

EXTRA = [{'model': sgd_regular.sgd_l1(l1=0.1, degree=2), 'name': 'l1=0.1 degree=2',
              'params': {}},
             {'model': sgd_regular.sgd_l2(l2=0.1, degree=2), 'name': 'l2=0.1 degree=2',
              'params': {}},
             {'model': sgd_regular.sgd_l1(l1=0.2, degree=3), 'name': 'l1=0.2 degree=3',
              'params': {}},
             {'model': sgd_regular.sgd_l2(l2=0.2, degree=3), 'name': 'l2=0.2 degree=3',
              'params': {}},
             {'model': sgd_regular.sgd_elastic(l1=0.1, l2=0.2, degree=2), 'name': 'l1=0.1 l2=0.2 degree=2',
              'params': {}},
             {'model': sgd_regular.sgd_elastic(l1=0.1, l2=0.2, degree=3), 'name': 'l1=0.1 l2=0.2 degree=3',
              'params': {}},
             ]

rows = 3
cols = 3


def test_linear():
    a = -100
    b = 1000
    n = 1000
    k = 10
    offset = 100
    test_name = f'linear {k}X+{offset}+eps'
    X = data.uniform(a, b, n)
    y = data.linear(X, k, offset, (b - a) / 100)
    mse_fig, res_fig = test.test_attrs(test_name, ATTRS, rows, cols, X, y)
    mse_fig.savefig('linear_mse.png')
    res_fig.savefig('linear_res.png')


def test_exp():
    a = 0
    b = 2
    n = 1000
    test_name = f'polynom'
    X = data.uniform(a, b, n)
    y = 3 ** X
    mse_fig, res_fig = test.test_attrs(test_name, ATTRS + EXTRA, rows + 1, cols + 1, X, y)
    mse_fig.savefig('exp_mse.png')
    res_fig.savefig('exp_res.png')

if __name__ == '__main__':
    X = data.uniform(-10, 20, 1000)
    y = data.linear(X, 3, 4, 5)
    test_linear()
    test_exp()
