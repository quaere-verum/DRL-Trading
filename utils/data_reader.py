import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from itertools import product
from torch.utils.data import Dataset

DATAFOLDER = os.environ['DATAFOLDER']

def get_ticker_data(assets, daterange, interval, fillna=True):
    '''
    :param assets: list of asset names to be retrieved (e.g. ['ETH', 'BTC']
    :param daterange: tuple of dates which serve as upper and lower bound, e.g. ('2020-01-01', '2021-01-01)
    :param interval: interval over which the data was collected.
    :param fillna: boolean which determines if 'Close' is frontfilled, and if 'Volume' is filled with zeros, if empty
    :return: dataframe with columns (coin, 'Close') and (coin, 'Volume') for coin in coins, for the specified interval\
    between the specified dates
    '''
    concat = pd.concat([pd.read_csv(fr'{DATAFOLDER}\\{asset}_full_{interval}.txt',
                          names=['timestamp', 'Close', 'Volume'], usecols=[0, 4, 5], index_col=0, parse_dates=[0])
              for asset in assets], axis=1).sort_index().asfreq('1h' if interval == '1hour' else interval).loc[daterange[0]:daterange[1]]
    if fillna:
        concat['Close'] = concat['Close'].ffill()
        concat['Volume'] = concat['Volume'].fillna(0)
    concat.columns = pd.MultiIndex.from_product([coins,['Close', 'Volume']])
    return concat

def default_target_function(data, lookahead_window=24, target_cols=None, q=0.9):
    '''
    :param lookahead_window: int that specifies how far into the future the reward is calculated
    :param target_cols: list of coin names whose future values are to be forecasted
    :param q: float between 0 and 1, determines the quantile to be forecasted
    :return: np.ndarray with targets. Make sure to drop all missing values.
    '''
    if target_cols is None:
        target_cols = list(data.columns.get_level_values(0).unique())
    return data.loc[:, [(col, 'Close') for col in target_cols]].sort_index(ascending=False).rolling(window=lookahead_window).apply(
                lambda x: np.quantile(x.values / x.values[-1] - 1, q)).sort_index().dropna().values

def prepare_data(data, lookback_window, features, test_size=0.05, shuffle=False, flatten=True, lookback_delta=1,
                 target_function=default_target_function):
    '''
    :param data: dataframe as returned from get_ticker_data
    :param lookback_window: int that specifies how far back into the past each observation sees
    :param features: dict of dictionaries, each having "function", "kwargs" and "gauge" as keys. "kwargs" must contain the\
    name of the feature and the keyword arguments of the feature's "function"
    :param test_size: size of test data
    :param shuffle: shuffle the dataset
    :param flatten: whether to reshape X to be of shape (len(X), -1)
    :param lookback_delta: delta of unit time per timestep in lookback window
    :param target_function: the function which takes data as input, and outputs the targets to be forecasted. The targets\
    must be based on future priceseries observations, and the function must return an np.ndarray with missing values dropped.\
    For example, if we output the maximum value over the future x timesteps, the final x-1 values should be dropped, and\
    remaining values should be returned as np.ndarray. See default_target_function for an example.
    :return: tuple of np.ndarrays, X_train, X_test, y_train, y_test
    Example:
        >> data = get_ticker_data(['ETH', 'BTC', 'ADA', 'TRX', 'XRP'], ('2020-01-01', '2023-01-01'), '1hour')
        >> from RL.feature_extraction.feature_functions import rolling_ou, rsi, rolling_quantile
        >> features = {'rsi': {'function': rsi, 'kwargs': {'n': 12, 'name': 'rsi_12'}, 'gauge': False},
                    'ou_est': {'function': rolling_ou, 'kwargs': {'clip_window': 48, 'est_window': 12, 'name': 'ou_12'}, 'gauge': True},
                    'quantile': {'function': rolling_quantile, 'kwargs': {'col': 'Close', 'window': 48, 'q': 0.5, 'name': 'q_48'}, 'gauge': True}}
        >> X_train, X_test, y_train, y_test = prepare_data(data, features=features)
    '''
    data = data.reset_index(drop=True)
    assets = list(data.columns.get_level_values(0).unique())
    futures = []
    with ProcessPoolExecutor(mp.cpu_count()) as executor:
        for (asset, feature) in product(assets, features.keys()):
            futures.append(executor.submit(features[feature]['function'], data=data[asset], asset=asset, **features[feature]['kwargs']))
    for future in futures:
        temp = future.result()
        data = pd.concat((data, temp), axis=1)
    data = data.dropna().reset_index(drop=True)
    data = data.sort_index(axis=1)
    targets = target_function(data)
    y = targets[lookback_window - 1::]
    covariates = np.stack(list(data.rolling(window=lookback_window))[lookback_window - 1::][0:len(y)], axis=0)
    if lookback_delta > 1:
        covariates = covariates[:, np.arange(start=covariates.shape[1]-1, stop=-1, step=-lookback_delta), :]
    gauge = data.columns.get_level_values(1).isin(['Close'] +
                  [features[feature]['kwargs']['name'] for feature in features if features[feature]['gauge']])
    conversion_matrix = np.stack([np.where(gauge & data.columns.get_level_values(0).isin([coin]), 1, 0)
                                  for coin in data.columns.get_level_values(0).unique()], axis=0)
    divisor = covariates[:, -1, np.arange(data.shape[1])[np.where(data.columns.get_level_values(1).isin(['Close']))]] @ conversion_matrix
    del data
    covariates = (covariates/np.where(divisor > 0, divisor, 1).reshape(-1, 1, len(gauge)))
    if flatten:
        covariates = covariates.reshape(len(covariates), -1)
    return train_test_split(covariates, y, test_size=test_size, shuffle=shuffle)


def data_writer(data, lookback_window, lookahead_window, features, folder_name, lookback_delta=1,
                 target_function=default_target_function):
    '''
    Implements a data writer which stores calculated features and labels in a .hdf5 file. It returns the necessary\
    arguments to construct a HDF5Dataset, which is an iterator that reads the data from the hard drive instead of\
    story it in working memory.
    :param data: dataframe as returned from get_ticker_data
    :param lookback_window: int that specifies how far back into the past each observation sees
    :param features: dict of dictionaries, each having "function", "kwargs" and "gauge" as keys. "kwargs" must contain the\
    name of the feature and the keyword arguments of the feature's "function"
    :param lookback_delta: delta of unit time per timestep in lookback window
    :param target_function: the function which takes data as input, and outputs the targets to be forecasted. The targets\
    must be based on future priceseries observations, and the function must return an np.ndarray with missing values dropped.\
    For example, if we output the maximum value over the future x timesteps, the final x-1 values should be dropped, and\
    remaining values should be returned as np.ndarray. See default_target_function for an example.
    :return: conversion_matrix and mask required to construct data loader
    Example:
        >> data = get_ticker_data(['ETH', 'BTC', 'ADA', 'TRX', 'XRP'], ('2020-01-01', '2023-01-01'), '1hour')
        >> from RL.feature_extraction.feature_functions import rolling_ou, rsi, rolling_quantile
        >> features = {'rsi': {'function': rsi, 'kwargs': {'n': 12, 'name': 'rsi_12'}, 'gauge': False},
                    'ou_est': {'function': rolling_ou, 'kwargs': {'clip_window': 48, 'est_window': 12, 'name': 'ou_12'}, 'gauge': True},
                    'quantile': {'function': rolling_quantile, 'kwargs': {'col': 'Close', 'window': 48, 'q': 0.5, 'name': 'q_48'}, 'gauge': True}}
        >> X_train, X_test, y_train, y_test = prepare_data(data, features=features)
    '''
    data = data.reset_index(drop=True)
    coins = list(data.columns.get_level_values(0).unique())
    futures = []
    with ProcessPoolExecutor(mp.cpu_count()) as executor:
        for (coin, feature) in product(coins, features.keys()):
            futures.append(executor.submit(features[feature]['function'], data=data[coin], asset=coin,
                                           **features[feature]['kwargs']))
    for future in futures:
        temp = future.result()
        data = pd.concat((data, temp), axis=1)
    data = data.dropna().reset_index(drop=True)
    data = data.sort_index(axis=1)
    targets = target_function(data)
    y = targets[lookback_window - 1::]
    data = data.iloc[0:-lookahead_window+1, :]
    if not os.path.exists(fr'{DATAFOLDER}\\{folder_name}'):
        os.makedirs(fr'{DATAFOLDER}\\{folder_name}')
    data.to_hdf(fr'{DATAFOLDER}\\{folder_name}\\data.hdf5', key='data')
    pd.DataFrame(y).to_hdf(fr'{DATAFOLDER}\\{folder_name}\\labels.hdf5', key='labels')
    gauge = data.columns.get_level_values(1).isin(['Close'] +
                                                  [features[feature]['kwargs']['name'] for feature in features if
                                                   features[feature]['gauge']])
    conversion_matrix = np.stack([np.where(gauge & data.columns.get_level_values(0).isin([coin]), 1, 0)
                                  for coin in data.columns.get_level_values(0).unique()], axis=0)
    mask = np.arange(data.shape[1])[np.where(data.columns.get_level_values(1).isin(['Close']))]

    return conversion_matrix, mask


class HDF5Dataset(Dataset):
    def __init__(self, path, lookback_window, conversion_matrix, mask, lookback_delta, in_memory=False):
        self.path = fr'{DATAFOLDER}\\{path}'
        self.lookback_window = lookback_window
        self.labels = pd.read_hdf(fr'{self.path}\labels.hdf5').values
        self.in_memory = in_memory
        if in_memory:
            self.data = pd.read_hdf(fr'{self.path}\data.hdf5').values
        self.conversion_matrix = conversion_matrix
        self.mask = mask
        self.lookback_ticks = np.arange(lookback_window-1, 0, -lookback_delta)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.in_memory:
            X = pd.read_hdf(fr'{self.path}\data.hdf5', start=idx, stop=idx+self.lookback_window).values
        else:
            X = self.data[idx:idx+self.lookback_window+1, :]
        divisor = X[-1, :][self.mask] @ self.conversion_matrix
        y = self.labels[idx]
        return X[self.lookback_ticks, :]/np.where(divisor > 0, divisor, 1), y


if __name__ == '__main__':
    lookback_window = 48
    lookahead_window = 48
    lookback_delta = 4
    data = get_ticker_data(['ETH', 'BTC', 'ADA', 'TRX', 'XRP'], ('2020-01-01', '2023-01-01'), '1hour')
    from RL.feature_extraction.feature_functions import rolling_ou, rsi, rolling_quantile
    features = {'rsi': {'function': rsi, 'kwargs': {'n': 12, 'name': 'rsi_12'}, 'gauge': False},
                'ou_est': {'function': rolling_ou, 'kwargs': {'clip_window': 48, 'est_window': 12, 'name': 'ou_12'}, 'gauge': True},
                'quantile': {'function': rolling_quantile, 'kwargs': {'col': 'Close', 'window': 48, 'q': 0.5, 'name': 'q_48'}, 'gauge': True}}
    target_function = lambda x: default_target_function(x, lookahead_window=48, target_cols=['ETH'], q=0.9)
    conversion_matrix, mask = data_writer(data, lookback_window, lookahead_window, features,
                                          'test', lookback_delta, target_function)
    in_memory = HDF5Dataset('test', lookback_window, conversion_matrix, mask, lookback_delta, True)
    harddrive = HDF5Dataset('test', lookback_window, conversion_matrix, mask, lookback_delta, False)
    import time
    from torch.utils.data import DataLoader
    in_memory_loader = DataLoader(in_memory, batch_size=64, shuffle=True, num_workers=mp.cpu_count())
    t = time.time()
    all = [x for x in in_memory_loader]
    print(f'Loading all data in memory: {time.time()-t}')
    harddrive_loader = DataLoader(harddrive, batch_size=64, shuffle=True, num_workers=mp.cpu_count())
    t = time.time()
    all = [x for x in harddrive_loader]
    print(f'Loading all data from harddrive: {time.time() - t}')
