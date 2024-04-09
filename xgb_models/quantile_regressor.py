from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import QuantileDMatrix
import numpy as np
import xgboost as xgb

class XGBRFQuantileRegressor:
    '''
    Turns the xgboost regressor into an ensemble model for the quantile error objective function, which is not supported\
    by xgboost itself. However, since this implementation uses the scikit-learn API, it is not possible to utilise\
    the DMatrix structure native to xgboost, which means training is substantially slower.
    '''
    def __init__(self, quantile_alpha, n_estimators, max_depth,
                 max_leaves=None, max_bin=None, grow_policy=None, learning_rate=0.4, verbosity=None, tree_method='hist',
                 n_jobs=1, gamma=None, min_child_weight=None, max_delta_step=None, subsample=None, colsample_bytree=None,
                 colsample_bylevel=None, colsample_bynode=None, reg_alpha=None, reg_lambda=None, scale_pos_weight=None,
                 random_state=None, max_samples=1., max_features=1., num_parallel_tree=1, early_stopping_rounds=None,
                 evals=None):
        if early_stopping_rounds is not None:
            raise NotImplementedError()
        self.xgb_regressor = XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile_alpha,
                                          n_estimators=n_estimators, max_depth=max_depth, device='cpu',
                 max_leaves=max_leaves, max_bin=max_bin, grow_policy=grow_policy, learning_rate=learning_rate,
                                          verbosity=verbosity, tree_method=tree_method, n_jobs=n_jobs, gamma=gamma,
                 min_child_weight=min_child_weight, max_delta_step=max_delta_step, subsample=subsample,
                                          colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
                 colsample_bynode=colsample_bynode, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                          scale_pos_weight=scale_pos_weight, random_state=random_state,
                                          early_stopping_rounds=early_stopping_rounds, evals=evals)
        self.early_stopping_rounds = early_stopping_rounds
        self.ensemble_model = BaggingRegressor(self.xgb_regressor, max_samples=max_samples, max_features=max_features,
                                               n_estimators=num_parallel_tree)

    def fit(self, X, y):
        self.ensemble_model.fit(X, y)

    def predict(self, X):
        return self.ensemble_model.predict(X)


class XGBCustomQuantileRegressor:
    '''
    The xgboost module does not support random forests with the quantile error objective function. This implementation\
    uses bootstrap aggregating (bagging), as well as feature bagging, to train a random forest to perform quantile\
    regression. Note: the DMatrix structure used to train xgboost models cannot be pickled. Hence, this class does\
    not support training multiple trees in parallel.
    '''
    def __init__(self, params, num_parallel_tree=1, features_per_tree=-1, samples_per_batch=1.):
        self.num_parallel_tree = num_parallel_tree
        self.features_per_tree = features_per_tree
        self.trees = dict()
        self.params = params
        self.params['objective'] = 'reg:quantileerror'
        self.samples_per_batch = samples_per_batch
        self.weights = None

    def train(self, X, y, val_data=None, early_stopping_rounds=None, num_boost_round=10):
        if early_stopping_rounds is not None:
            assert val_data is not None, 'Validation data is required to be able to set early stopping.'
        for k in range(self.num_parallel_tree):
            X_batch, y_batch, features = self.prepare_batch(X, y, val_data)
            dtrain = QuantileDMatrix(X_batch, label=y_batch)
            del X_batch, y_batch
            evals = None
            if val_data is not None:
                evals = [(QuantileDMatrix(val_data[0][:, features], label=val_data[1]), 'dtest')]
            model = xgb.train(dtrain=dtrain, params=self.params, evals=evals, num_boost_round=num_boost_round,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

            self.trees[f'tree_{k}'] = {'model': model, 'features': features}
            del model, features, dtrain
        if val_data is not None:
            scores = np.array([self.trees[tree]['model'].best_score for tree in self.trees])
            self.weights = np.exp(scores) / np.sum(np.exp(scores)).reshape(1, 1, -1)

    def prepare_batch(self, X, y, val_data):
        batch_size = int(len(X) * self.samples_per_batch)
        batch_ind = np.random.choice(np.arange(len(X)), batch_size)
        if self.features_per_tree == -1:
            features = np.arange(X.shape[1])
            X_batch = X[batch_ind][:, features]
            y_batch = y[batch_ind]
        else:
            features = np.random.choice(np.arange(X.shape[1]), self.features_per_tree, replace=False)
            X_batch = X[batch_ind][:, features]
            y_batch = y[batch_ind]
        return X_batch, y_batch, features

    def predict(self, X):
        preds = np.zeros((len(X), len(self.params['quantile_alpha']), self.num_parallel_tree))
        for k, tree in enumerate(self.trees):
            qmat = QuantileDMatrix(X[:, self.trees[tree]['features']])
            preds[:, :, k] = self.trees[tree]['model'].predict(qmat)
        if self.weights is not None:
            return np.sum(self.weights * preds, axis=-1)
        return np.mean(preds, axis=-1)

class XGBTemporalQuantileRegressor(XGBCustomQuantileRegressor):
    '''
    The super class XGBCustomQuantileRegressor trains a random forest to perform quantile regression. One of the crucial\
    parts of training a random forest is bootstrap aggregating, which entails training random trees on random batches\
    with randomly selected features. If our input is a multivariate timeseries, however, we may wish to preserve some\
    characteristics of the timeseries. This model will randomly select timesteps between 0 and the lookback window, and\
    randomly select columns from the multivariate timeseries. But we then get the selected features from each\
    of the selected timesteps, as opposed to getting a completely random bucket of features from the entire flattened\
    timeseries, which is what the super class would give us.
    '''
    def __init__(self, params, input_shape, num_parallel_tree=1, timesteps_per_tree=-1,
                 features_per_timestep=-1, samples_per_batch=1.):
        super().__init__(params, num_parallel_tree, timesteps_per_tree*features_per_timestep, samples_per_batch)
        self.timesteps_per_tree = timesteps_per_tree
        self.features_per_timestep = features_per_timestep
        self.input_shape = input_shape

    def prepare_batch(self, X, y, val_data):
        batch_size = int(len(X) * self.samples_per_batch)
        batch_ind = np.random.choice(np.arange(len(X)), batch_size)
        if self.timesteps_per_tree == -1:
            if self.features_per_timestep == -1:
                features = np.arange(X.shape[1])
            else:
                timeseries_features = np.random.choice(np.arange(self.input_shape[1]), self.features_per_timestep, replace=False)
                features = (self.input_shape[1]*np.arange(self.input_shape[0]).reshape(-1, 1) + timeseries_features).reshape(-1)
        else:
            timesteps = np.random.choice(np.arange(self.input_shape[0]), self.timesteps_per_tree, replace=False)
            if self.features_per_timestep == -1:
                features = (self.input_shape[1]*timesteps.reshape(-1, 1) + np.arange(self.input_shape[1])).reshape(-1)
            else:
                timeseries_features = np.random.choice(np.arange(self.input_shape[1]), self.features_per_timestep, replace=False)
                features = (self.input_shape[1]*timesteps.reshape(-1, 1) + timeseries_features).reshape(-1)
        X_batch = X[batch_ind][:, features]
        y_batch = y[batch_ind]
        return X_batch, y_batch, features


#Short example to illustrate the difference
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from RL.utils.data_reader import get_ticker_data, prepare_data, default_target_function
    import matplotlib.pyplot as plt
    import time
    from RL.feature_extraction.feature_functions import rolling_ou, rsi, rolling_quantile

    np.random.seed(123)
    coins = ['BTC', 'ETH', 'XRP', 'ADA']
    data = get_ticker_data(coins=coins, daterange=('2020-01-01', '2023-01-01'), interval='1hour', fillna=True).reset_index(drop=True)
    lookback_window = 48
    lookahead_window = 24

    features = {'rsi': {'function': rsi, 'kwargs': {'n': 12, 'name': 'rsi_12'}, 'gauge': False},
                'ou_est': {'function': rolling_ou, 'kwargs': {'clip_window': 48, 'est_window': 12, 'name': 'ou_12'},
                           'gauge': True},
                'quantile': {'function': rolling_quantile,
                             'kwargs': {'col': 'Close', 'window': 48, 'q': 0.5, 'name': 'q_48'}, 'gauge': True}}
    X_train, X_test, y_train, y_test = prepare_data(data, 48, features, 0.02, False,
                                                    True, 5, default_target_function)
    print('Data preparation done. Training models.')

    params = {'max_depth': 10, 'quantile_alpha': np.array([0.1, 0.5, 0.9])}
    timeseries_model = XGBTemporalQuantileRegressor(params, (lookback_window, 2*len(coins)), 250,
                                                    lookback_window // 4, 4, 0.75)
    timeseries_start = time.time()
    timeseries_model.train(X_train, y_train, val_data=(X_test, y_test), early_stopping_rounds=2, num_boost_round=15)
    timeseries_time = time.time() - timeseries_start
    timeseries_preds = timeseries_model.predict(X_test)

    base_model = XGBRegressor(objective='reg:quantileerror', max_depth=6, n_estimators=15,
                              quantile_alpha=np.array([0.1, 0.5, 0.9]), num_parallel_tree=1)
    base_start = time.time()
    base_model.fit(X_train, y_train)
    base_time = time.time() - base_start
    base_preds = base_model.predict(X_test)

    ensemble_model = XGBRFQuantileRegressor(max_depth=10, n_estimators=15, quantile_alpha=np.array([0.1, 0.5, 0.9]),
                     num_parallel_tree=250, max_samples=0.75, max_features=X_train.shape[1] // 2, early_stopping_rounds=None)
    ensemble_start = time.time()
    ensemble_model.fit(X_train, y_train.reshape(-1))
    ensemble_time = time.time() - ensemble_start
    ensemble_preds = ensemble_model.predict(X_test)

    params = {'max_depth': 10, 'quantile_alpha': np.array([0.1, 0.5, 0.9])}
    custom_model = XGBCustomQuantileRegressor(params, 250, X_train.shape[1]//2, 0.75)
    custom_start = time.time()
    custom_model.train(X_train, y_train, val_data=(X_test, y_test), early_stopping_rounds=2, num_boost_round=25)
    custom_time = time.time() - custom_start
    custom_preds = custom_model.predict(X_test)


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    ax[0, 0].plot(y_test, c='green')
    ax[0, 0].plot(base_preds[:, 1], c='red')
    ax[0, 0].plot(base_preds[:, 0], c='blue', alpha=0.5)
    ax[0, 0].plot(base_preds[:, 2], c='purple', alpha=0.5)
    ax[0, 0].set_title(f'Random Tree ({base_time:.3f}s)')

    ax[0, 1].plot(y_test, c='green')
    ax[0, 1].plot(ensemble_preds[:, 1], c='red')
    ax[0, 1].plot(ensemble_preds[:, 0], c='blue', alpha=0.5)
    ax[0, 1].plot(ensemble_preds[:, 2], c='purple', alpha=0.5)
    ax[0, 1].set_title(f'Random Forest ({ensemble_time:.3f}s)')

    ax[1, 0].plot(y_test, c='green')
    ax[1, 0].plot(custom_preds[:, 1], c='red')
    ax[1, 0].plot(custom_preds[:, 0], c='blue', alpha=0.5)
    ax[1, 0].plot(custom_preds[:, 2], c='purple', alpha=0.5)
    ax[1, 0].set_title(f'Custom Forest ({custom_time:.3f}s)')

    ax[1, 1].plot(y_test, c='green')
    ax[1, 1].plot(timeseries_preds[:, 1], c='red')
    ax[1, 1].plot(timeseries_preds[:, 0], c='blue', alpha=0.5)
    ax[1, 1].plot(timeseries_preds[:, 2], c='purple', alpha=0.5)
    ax[1, 1].set_title(f'Temporal Forest ({timeseries_time:.3f}s)')

    plt.show()

