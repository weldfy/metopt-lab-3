import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


def maximize_train(sgd, X_train, X_test, y_train, y_test, batch_sizes, learning_rates, epochs, params):
    best = None
    for epoch in epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                sgd.train(X_train, y_train, batch_size, learning_rate, epoch, True, **params)
                y_pred = sgd.predict(X_test)
                mse = utils.mse(y_test, y_pred)
                if best is None or mse < best[-1]:
                    best = (batch_size, learning_rate, epoch, mse)
    sgd.train(X_train, y_train, best[0], best[1], best[2], True, **params)
    return sgd


def test_attrs(title, models, rows, cols, X, y):
    batch_sizes = [16, 32]
    learning_rates = [0.001, 0.01, 0.1]
    epochs = [10, 50, 100]
    fig1, axes1 = plt.subplots(rows, cols, figsize=(24, 12))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(24, 12))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    fig1.suptitle(f'{title} MSE LOSS')
    fig2.suptitle(f'{title} RESULT')
    for i, model in enumerate(models):
        sgd = maximize_train(model['model'], X_train, X_test, y_train, y_test, batch_sizes, learning_rates, epochs, model['params'])
        title = model['name']
        row = i // cols
        col = i - row * cols
        mse = axes1[row, col]
        res = axes2[row, col]
        epoch_info(title, sgd, mse)
        result_info(title, sgd, X_test, y_test, res)
    plt.show()
    return fig1, fig2

def epoch_info(title, sgd, ax):
    epochs = sgd.info['epochs']
    mse_df = pd.DataFrame({'epochs': [i for i in range(epochs)], 'mse': sgd.info['mse'][:epochs]})
    ax.set_title(f'{title}')
    sns.scatterplot(mse_df, x='epochs', y='mse', ax=ax)


def result_info(title, sgd, X_test, y_test, ax):
    epochs = sgd.info['epochs']
    X = X_test.ravel()
    y = y_test.ravel()
    ax.set_title(f'{title}')
    df = pd.DataFrame({'X': X, 'y': y})
    sns.lineplot(df, x='X', y='y', ax=ax)

    df = pd.DataFrame({'X': X, 'y': sgd.predict(X_test).ravel()})
    sns.lineplot(df, x='X', y='y', ax=ax)
