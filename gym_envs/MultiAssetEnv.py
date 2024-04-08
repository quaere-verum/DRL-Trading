import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import warnings
from itertools import product
from RL.gym_envs.utils.history import History
from RL.gym_envs.utils.portfolio import Portfolio

warnings.filterwarnings("error")

def basic_reward_function(info):
    val = info['portfolio_valuation', -1] / info['portfolio_valuation', -2]
    return np.log(max(val, 1e-8))


class MultiAssetEnv(gym.Env):
    '''
    Must be registered via gymnasium.envs.register for proper usage as a gym environment.
    '''
    metadata = {'render_modes': ['logs', 'human']}
    def __init__(self, data, assets, feature_properties={}, reward_function=basic_reward_function, gauge_asset=None,
                 episode_length=None, capital=100, dynamic_features={},
                 max_invest=10, lookback_window=10, lookahead_window=10, print_interval=100,
                 render_mode='logs', verbose=1):

        self.base_data = data.reset_index(drop=True)
        self.assets = assets
        self.reward_function = reward_function
        self.feature_properties = feature_properties
        self.dynamic_features = dynamic_features
        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window
        self._set_data()
        assert (len(self.other_features) > 0 and len(self.sequential_features) > 0), \
            ('Environment is designed to contain both sequential features and non-sequential features.')
        self._calc_rewards()

        self.initial_capital = capital
        self.max_invest = max_invest
        self.episode_length = episode_length
        self.gauge_asset = gauge_asset if gauge_asset is not None else assets[0]

        self.action_space = spaces.Box(0, 1, shape=(len(self.assets),), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {'prices': spaces.Box(-np.inf, np.inf, shape=(self.lookback_window, self.prices.shape[1]),
                                  dtype=np.float32),
             'sequential': spaces.Box(-np.inf, np.inf, shape=(self.lookback_window, len(self.sequential_features)),
                                      dtype=np.float32),
             'other': spaces.Box(-np.inf, np.inf, shape=(len(self.other_features),), dtype=np.float32)}
        )

        self.print_interval = print_interval
        self.reward_history = {'reward': [], 'profit': []}
        self.render_mode = render_mode
        self.verbose = verbose

    def _set_data(self):
        self.features_frame = self.base_data[[(asset, feature)
                                              for asset, feature in
                                              product(self.assets, self.feature_properties)]].copy()
        for dynamic_feature in self.dynamic_features:
            self.features_frame[('temp', dynamic_feature)] = self.dynamic_features[dynamic_feature]['initial_value']
        self.features_frame = self.features_frame.astype(np.float32)

        self.sequential_features = [(asset, feature) for asset, feature in product(self.assets, self.feature_properties)
                                    if self.feature_properties[feature]['sequential']]
        self.sequential_features.extend(
            [('temp', name) for name in self.dynamic_features if self.dynamic_features[name]['sequential']])

        self.gauge = [self.feature_properties[feature]['gauge'] for _, feature in
                      product(self.assets, self.feature_properties)
                      if self.feature_properties[feature]['sequential']]
        self.gauge.extend([self.dynamic_features[name]['gauge'] for name in self.dynamic_features if
                           self.dynamic_features[name]['sequential']])

        self.other_features = [(asset, feature) for asset, feature in product(self.assets, self.feature_properties)
                               if not self.feature_properties[feature]['sequential']]
        self.other_features.extend(
            [('temp', name) for name in self.dynamic_features if not self.dynamic_features[name]['sequential']])

        self.prices = self.base_data[[(asset, 'Close') for asset in self.assets]].astype(np.float32)

    def _calc_rewards(self):
        temp = self.base_data[[(asset, 'Close') for asset in self.assets]].copy().sort_index(ascending=False)
        max_rise = temp.rolling(window=self.lookahead_window).apply(lambda x: np.max(x.values / x.values[0])) - 1
        max_fall = temp.rolling(window=self.lookahead_window).apply(lambda x: np.min(x.values / x.values[0])) - 1
        max_rise.columns = pd.MultiIndex.from_tuples([(asset, 'max_rise') for asset in self.assets])
        max_fall.columns = pd.MultiIndex.from_tuples([(asset, 'max_fall') for asset in self.assets])
        self.rewards_table = pd.concat((max_rise, max_fall), axis=1).sort_index().dropna()

    def _get_targets(self, positions):
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._idx = self.lookback_window - 1
        self._position = np.zeros(len(self.assets)).astype(np.float32)
        self._portfolio = Portfolio(self.initial_capital, self.lookahead_window, self.max_invest)

        self.info = History(len(self.base_data))
        self.info.set(
            idx=self._idx,
            capital=self.initial_capital,
            portfolio_valuation=np.float32(self.initial_capital),
            target_actions=np.zeros(len(self.assets)).astype(np.float32)
        )
        self.total_reward = 0
        return self.get_obs(), self.info[0]

    def step(self, action):
        self._position = action
        self._expected_return = self.features_frame.loc[self._idx, [(asset, 'pred') for asset in self.assets]].values
        self._idx += 1
        self._step += 1

        prices = dict(zip(self.assets, self.prices.iloc[self._idx, :].values))
        self._portfolio._trade_positions(prices, self._position, self._expected_return)
        self.info.add(
            idx=self._idx,
            capital=self._portfolio.capital,
            portfolio_valuation=self._portfolio.valuation(prices),
            target_actions=self._get_targets(self._position)
        )

        done = self.info['portfolio_valuation', -1] <= 0.5
        truncated = (self._idx >= len(self.base_data) - 1 - self.lookahead_window) or (
                    self._step == self.episode_length)
        if not done and not truncated:
            reward = self.reward_function(self.info)
            self.total_reward += reward
        else:
            profit = self.info['portfolio_valuation'] / self.initial_capital - 1
            if self.verbose > 0:
                std = np.std(profit)
                if std == 0:
                    print('Warning: agent remained passive the entire episode.')
                    std = 1
                print('Sharpe ratio: ', profit[-1] / std)
            reward = 0
            self.reward_history['reward'].append(self.total_reward)
            self.reward_history['profit'].append(profit)
        return self.get_obs(), reward, done, truncated, self.info[-1]

    def get_obs(self):
        for feature in self.dynamic_features:
            self.features_frame.loc[self._idx, ('temp', feature)] = self.dynamic_features[feature]['function'](
                self.info)
        prices = self.prices.loc[self._idx - self.lookback_window + 1:self._idx, :].values
        sequential_features = self.features_frame.loc[self._idx - self.lookback_window + 1:self._idx,
                              self.sequential_features].values
        seq_divisor = np.where(self.gauge, self.prices.loc[self._idx, (self.gauge_asset, 'Close')], 1).astype(
            np.float32)
        features = self.features_frame.loc[self._idx, self.other_features].values
        return {'prices': prices / self.prices.loc[self._idx, (self.gauge_asset, 'Close')],
                'sequential': sequential_features / seq_divisor,
                'other': features}

    def render(self):
        pass


class MultiAssetEnvDiscrete(MultiAssetEnv):
    '''
    Subclass of MultiAssetEnv whose action space is {0, 1}. The action 1 corresponds to following the "advice" that is\
    given by the column (asset, pred). This means that, for each timestep, if 0 is chosen, no funds will be invested. If\
    1 is chosen, we invest max_invest/n_assets into each asset, and we expect to be able to make a return of (asset, pred)\
    within lookahead_window timesteps. If this does not happen, or if a stop loss threshold is reached, the position is\
    closed.
    '''
    metadata = {'render_modes': ['logs', 'human']}

    def __init__(self, data, assets, feature_properties={}, reward_function=basic_reward_function, gauge_asset=None,
                 episode_length=None, capital=100, dynamic_features={},
                 max_invest=10, lookback_window=10, lookahead_window=10, print_interval=100,
                 render_mode='logs', verbose=1):
        super().__init__(data=data, assets=assets, feature_properties=feature_properties, reward_function=reward_function,
                         gauge_asset=gauge_asset, episode_length=episode_length, capital=capital,
                         dynamic_features=dynamic_features, max_invest=max_invest, lookback_window=lookback_window,
                         lookahead_window=lookahead_window, print_interval=print_interval, render_mode=render_mode,
                         verbose=verbose)
        self.action_space = spaces.Discrete(n=2, start=0)


class MultiAssetEnvMultiDiscrete(MultiAssetEnv):
    '''
    Subclass of MultiAssetEnv whose action space is {0, 1} x ... x {0, 1}. The action 1 corresponds to following the\
    "advice" that is given by the column (asset, pred). This means that, for each timestep, if 0 is chosen, no funds will
    be invested into the respective asset. If 1 is chosen, we invest max_invest/n_assets into the asset, and we expect to\
    be able to make a return of (asset, pred) within lookahead_window timesteps. If this does not happen, or if a stop\
    loss threshold is reached, the position is closed.
    '''
    metadata = {'render_modes': ['logs', 'human']}

    def __init__(self, data, assets, feature_properties={}, reward_function=basic_reward_function, gauge_asset=None,
                 episode_length=None, capital=100, dynamic_features={},
                 max_invest=10, lookback_window=10, lookahead_window=10, print_interval=100,
                 render_mode='logs', verbose=1):
        super().__init__(data=data, assets=assets, feature_properties=feature_properties,
                         reward_function=reward_function,
                         gauge_asset=gauge_asset, episode_length=episode_length, capital=capital,
                         dynamic_features=dynamic_features, max_invest=max_invest, lookback_window=lookback_window,
                         lookahead_window=lookahead_window, print_interval=print_interval, render_mode=render_mode,
                         verbose=verbose)
        self.action_space = spaces.MultiDiscrete(nvec=np.ones(len(assets))*2)

if __name__ == '__main__':
    # Generate synthetic data and go through an episode with random actions.
    np.random.seed(123)
    assets = ['ETH-USD', 'BTC-USD']
    data = pd.DataFrame(np.exp(np.cumsum(np.random.normal(scale=0.005, size=(100, 6)), axis=0)))
    data.columns = pd.MultiIndex.from_tuples([(asset, 'Close') for asset in assets]
                                             + [(asset, 'ou_60') for asset in assets]
                                             + [(asset, 'pred') for asset in assets])
    dynamic_features = {'capital': {'sequential': False, 'gauge': False, 'initial_value': 100,
                                    'function': lambda x: np.float32(x['capital', -1])}}
    print('MultiAssetEnv:')
    test_env = MultiAssetEnv(data, assets, feature_properties={'ou_60': {'sequential': True, 'gauge': True},
                                                           'pred': {'sequential': True, 'gauge': False}},
                         dynamic_features=dynamic_features)
    obs, info = test_env.reset()
    terminated = False
    while not terminated:
        positions = np.random.uniform(0, 1, size=len(assets) + 1)
        obs, reward, done, truncated, info = test_env.step(positions)
        terminated = done or truncated

    print('MultiAssetEnvDiscrete:')
    discrete_test_env = MultiAssetEnvDiscrete(data, assets, feature_properties={'ou_60': {'sequential': True, 'gauge': True},
                                                               'pred': {'sequential': True, 'gauge': False}},
                             dynamic_features=dynamic_features)
    obs, info = discrete_test_env.reset()
    terminated = False
    while not terminated:
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = discrete_test_env.step(action)
        terminated = done or truncated

    print('MultiAssetEnvMultiDiscrete:')
    multi_discrete_test_env = MultiAssetEnvDiscrete(data, assets,
                                              feature_properties={'ou_60': {'sequential': True, 'gauge': True},
                                                                  'pred': {'sequential': True, 'gauge': False}},
                                              dynamic_features=dynamic_features)
    obs, info = multi_discrete_test_env.reset()
    terminated = False
    while not terminated:
        action = np.random.randint(0, 2, size=len(assets))
        obs, reward, done, truncated, info = multi_discrete_test_env.step(action)
        terminated = done or truncated
