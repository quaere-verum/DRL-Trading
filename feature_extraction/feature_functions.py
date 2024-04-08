import scipy.stats as ss
import numpy as np
import pandas as pd

'''
All feature functions must be defined in a specific way. This is so that they can be passed to the prepare_data function,
which uses the multiprocessing module to calculate the features in parallel for multiple assets. This is why "asset" and\
"name" must be passed as arguments, and the result must return a pd.Series or pd.DataFrame, whose name (resp. columns) is\
(asset, name) (resp. product([asset], name)).
'''

def rsi(data, n, asset=None, name=None):
    '''
    Standard RSI calculation.
    :param data: dataframe containing the column "Close"
    :param n: window across which to calculate RSI
    :param asset: name of the asset whose RSI is calculated
    :param name: name of the returned series (e.g. "rsi_{n}")
    :return:
    '''
    s = np.array(data['Close']).reshape(-1)
    U = pd.Series(np.clip(s[1::]-s[0:-1], 0, np.inf)).dropna()
    D = pd.Series(np.clip(-(s[1::]-s[0:-1]), 0, np.inf)).dropna()
    U_rolling = U.rolling(window=n).mean()
    D_rolling = D.rolling(window=n).mean()
    return pd.Series(U_rolling/(U_rolling+D_rolling), name=(asset, name))

def rolling_ou(data, clip_window=1000, est_window=300, asset=None, name=None):
    '''
    Fits an Ornstein-Uhlenbeck process to each window (consisting of est_window timesteps) and returns the estimated\
    mean to which the process is reverting. Subsequently clips the value to prevent egregious estimates.
    :param data: dataframe containing the column "Close"
    :param clip_window: window across which to look back to clip the output value
    :param est_window: window across which to estimate the Ornstein-Uhlenbeck process
    :param asset: name of the asset whose rolling Ornstein-Uhlenbeck parameters are being calculated
    :param name: name of the returned series (e.g. "ou_{est_window})"
    :return: pd.Series containing rolling Ornstein-Uhlenbeck estimates
    '''
    return pd.Series(data['Close'].rolling(window=clip_window).apply(lambda x: np.clip(mu_est(x[-est_window::]),
        x.values[-1] - 0.5*np.abs(np.max(x-x.values[-1])), x.values[-1] + 0.5*np.abs(np.max(x-x.values[-1])))), name=(asset, name))

def mu_est(s):
    '''
    :param s: iterable (most likely pd.Series, np.ndarray or list)
    :return: the OLS estimated mean to which the Ornstein-Uhlenbeck process fitted to s is reverting
    '''
    s = np.array(s).reshape(-1)
    if np.std(s) == 0:
        return 0
    XX = s[:-1]
    YY = s[1:]
    try:
        beta, alpha, _, _, _ = ss.linregress(XX, YY)
    except:
        return np.nan
    mu_ols = alpha / (1 - beta)
    return mu_ols

def rolling_mean(data, col, window, asset=None, name=None):
    return pd.Series(data[col].rolling(window=window).mean(), name=(asset, name))

def rolling_std(data, col, window, asset=None, name=None):
    return pd.Series(data[col].rolling(window=window).std(), name=(asset, name))

def bollinger_bands(data, window, asset=None, name=None):
    upper = pd.Series(data['Close'].rolling(window=window).mean() + data['Close'].rolling(window=window).std(),
                      name=(asset, f'{name}_upper'))
    lower = pd.Series(data['Close'].rolling(window=window).mean() - data['Close'].rolling(window=window).std(),
                      name=(asset, f'{name}_lower'))
    return pd.concat((upper, lower), axis=1)

def rolling_quantile(data, col, window, q=0.5, asset=None, name=None):
    return pd.Series(data[col].rolling(window=window).quantile(q), name=(asset, name))

def fourier_reduction(s, drop=(1, -1)):
    '''
    Calculate the Fourier transform of a timeseries "s", drop the frequencies specified by the parameter "drop".
    :param s: iterable, typically pd.Series or np.ndarray
    :param drop: tuple or list. If drop is a tuple, all frequency contributions between drop[0] and drop[1] will be dropped.
    :return: transformed timeseries as np.ndarray
    '''
    ft = np.fft.fft(np.array(s).reshape(-1))
    if isinstance(drop, tuple):
        ft[drop[0]:drop[1]] = 0
    else:
        ft[drop] = 0
    return np.real(np.fft.ifft(ft))

def rolling_fourier_transform(data, window, asset=None, name=None):
    pass