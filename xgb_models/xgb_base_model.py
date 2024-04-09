from xgboost import DMatrix
import numpy as np
import xgboost as xgb

class XGBCustomRegressor:
    def __init__(self, params, objective=None):
        self.params = params
        self.objective = objective
        if objective is not None:
            assert 'objective' not in params.keys(), 'Either use the params to set the objective, or the keyword argument.'
        self.model = None

    def train(self, X, y, val_data=None, early_stopping_rounds=None, num_boost_round=10, verbose_eval=False):
        if early_stopping_rounds is not None:
            assert val_data is not None, 'Validation data is required to be able to set early stopping.'
        if isinstance(X, DMatrix):
            assert y is None
            dtrain = X
        else:
            dtrain = DMatrix(X, label=y)
        evals = None
        if val_data is not None:
            if isinstance(val_data, DMatrix):
                evals = [(val_data, 'dtest')]
            else:
                evals = [(DMatrix(val_data[0], label=val_data[1]), 'dtest')]
        self.model = xgb.train(dtrain=dtrain, params=self.params, evals=evals, num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval, obj=self.objective)


    def predict(self, X, output_margin=False):
        return self.model.predict(DMatrix(X), output_margin=output_margin)

def custom_loss(y_hat, dtrain, mult=5, eps = 0.005):
    '''
    Implementation of custom loss function L(y, y_hat) = (y-y_hat)**2 if |y_hat - y| > eps else mult*(y-y_hat)**2.\
    Concretely, this means we are trying to squeeze the predictions to lie within a strip of width 2*eps around the label.\
    :param y_hat: model predictions
    :param dtrain: DMatrix which contains the labels y=dtrain.get_label()
    :param mult:
    :param eps:
    :return:
    '''
    y = dtrain.get_label()
    condition = np.abs(y_hat - y) > eps
    grad = -np.where(condition, mult*2.*(y-y_hat), 2*(y-y_hat))
    hess = -np.where(condition, -2.*mult, -2.)
    return grad, hess

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from RL.utils.data_reader import get_ticker_data, prepare_data
    import matplotlib.pyplot as plt
    import time
    from RL.feature_extraction.feature_functions import rsi, rolling_mean, bollinger_bands, rolling_std
    import multiprocessing

    np.random.seed(123)
    coins = ['ETH', 'BTC']
    data = get_ticker_data(coins=coins, daterange=('2019-01-01', '2023-08-01'), interval='5min', fillna=True).reset_index(drop=True)
    lookback_window = 240
    lookahead_window = 48

    features = {'rsi': {'function': rsi, 'kwargs': {'n': 20, 'name': 'rsi_20'}, 'gauge': False},
                'mean': {'function': rolling_mean,
                        'kwargs': {'col': 'Close', 'window': 48, 'name': 'mean_48'}, 'gauge': True},
                'bollinger_bands': {'function': bollinger_bands,
                         'kwargs': {'window': 48, 'name': 'bollinger_band_48'}, 'gauge': True}
                }


    def target_function(data, target_cols, min_profit=0., max_profit=0.04, lookahead_window=24, q=0.9):
        temp = data.loc[:, [(col, 'Close') for col in target_cols]].sort_index(ascending=False).rolling(
            window=lookahead_window).apply(
            lambda x: np.quantile(x.values / x.values[-1] - 1, q)).sort_index().dropna().values
        return np.clip(temp, min_profit, max_profit)

    X_train, X_test, y_train, y_test = prepare_data(data, lookback_window, features,
                                        0.01, False, True, 2,
                                                    lambda x: target_function(x, ['ETH']))
    dtrain = DMatrix(X_train, label=y_train, nthread=multiprocessing.cpu_count())
    dtest = DMatrix(X_test, label=y_test, nthread=multiprocessing.cpu_count())
    print('Data preparation done. Training models.')

    base_params = {'max_depth': 6, 'device': 'cuda', 'objective': 'reg:squarederror'}
    base_model = XGBCustomRegressor(base_params, None)
    base_start = time.time()
    base_model.train(X=dtrain, y=None, val_data=dtest, verbose_eval=True)
    base_time = time.time() - base_start
    base_preds = base_model.predict(X_test, False)

    custom_params = {'max_depth': 10, 'device': 'cuda', 'eval_metric': 'rmse',
                      'max_bin': 256, 'eta': 0.2}
    custom_model = XGBCustomRegressor(custom_params, custom_loss)
    custom_start = time.time()
    custom_model.train(X=dtrain, y=None, val_data=dtest, num_boost_round=40, early_stopping_rounds=2,
                       verbose_eval=True)
    custom_time = time.time() - custom_start
    custom_preds = custom_model.predict(X_test, False)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot(y_test, c='green', label='target')
    ax.plot(base_preds, c='red', label=f'base ({base_time:.3f}s)', alpha=0.5)
    ax.plot(custom_preds, c='blue', label=f'custom ({custom_time:.3f}s)', alpha=0.5)
    ax.legend()


    plt.show()