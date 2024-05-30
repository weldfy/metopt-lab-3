import sgd
import os

import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras

MODEL_OPT = {
    'SGD': keras.optimizers.SGD,
    'Nesterov': keras.optimizers.SGD,
    'Adagrad': keras.optimizers.Adagrad,
    'RMSprop': keras.optimizers.RMSprop,
    'Adam': keras.optimizers.Adam
}


class SGDKeras(sgd.SGD):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.opt = MODEL_OPT[model_name]
        self.nesterov = False
        if self.model_name == 'Nesterov':
            self.nesterov = True
        self.model = None

    def fit(self, X, y, batch_size, learning_rate, epochs, calc_metrics, **kwargs):
        self.model = keras.Sequential()
        _, n = X.shape
        self.model.add(keras.layers.Dense(1, input_shape=(n,)))
        if self.nesterov:
            kwargs['nesterov'] = True
        self.model.compile(optimizer=self.opt(learning_rate=learning_rate, **kwargs),
                           loss='mean_squared_error')
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        if calc_metrics:
            self.info['mse'] = history.history['loss']

    def scale(self, data):
        return data

    def rescale(self, data):
        return data

    def predict(self, X):
        return self.model.predict(X)