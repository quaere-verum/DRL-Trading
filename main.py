import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import warnings
from stable_baselines3 import PPO
import torch as th
from feature_extraction.torch_modules import CombinedLSTMExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
warnings.filterwarnings(category=UserWarning, action='ignore')

gym.envs.register('MultiAssetEnv', 'RL.gym_envs.MultiAssetEnv:MultiAssetEnv')
gym.envs.register('MultiAssetEnvDiscrete', 'RL.gym_envs.MultiAssetEnv:MultiAssetEnvDiscrete')
gym.envs.register('MultiAssetEnvMultiDiscrete', 'RL.gym_envs.MultiAssetEnv:MultiAssetEnvMultiDiscrete')

def fake_priceseries_generator(n_steps, n_assets, std=0.005):
    covariance = np.random.uniform(low=-1, high=1, size=(n_assets, n_assets))
    covariance = (covariance.T+covariance)/2
    np.fill_diagonal(covariance, 1)
    increments = np.random.multivariate_normal(np.zeros(n_assets), covariance, n_steps)
    return pd.DataFrame(np.exp(np.cumsum(increments, axis=0)*std), columns=
    pd.MultiIndex.from_tuples([(f'asset_{k}', 'Close') for k in range(n_assets)]))

def fake_predictions(data, window, quantile, rmse=0.01):
    temp = data.copy().sort_index(ascending=False).rolling(window=window).apply(lambda x: \
                                                        np.quantile(x.values/x.values[-1]-1, quantile))
    temp.columns = pd.MultiIndex.from_tuples([(asset, 'pred') for asset in data.columns.get_level_values(0)])
    return temp.sort_index()+np.random.normal(0, rmse/2, size=temp.shape)

def main():
    initial_capital = 100
    n_assets = 4
    n_steps = 200
    lookback_window = 25
    prices = fake_priceseries_generator(n_steps, n_assets, 0.01)
    preds = fake_predictions(prices, 15, 0.5)
    data = pd.concat((prices, preds), axis=1).dropna().reset_index(drop=True)
    feature_properties={
        'pred':{
            'sequential': True,
            'gauge': False
        }
    }
    dynamic_features={
        'capital':{
            'sequential': False,
            'initial_value': initial_capital,
            'function': lambda x: np.float32(x['capital', -1])
        }
    }
    env = make_vec_env('MultiAssetEnv', env_kwargs={'data': data,
                                                    'assets': prices.columns.get_level_values(0),
                                                    'capital': initial_capital,
                                                    'lookback_window': lookback_window,
                                                    'feature_properties': feature_properties,
                                                    'dynamic_features': dynamic_features,
                                                    'verbose': 0,
                                                    'render_mode': 'logs'}, n_envs=8)
    test_env = gym.make('MultiAssetEnvDiscrete', data=data, assets=prices.columns.get_level_values(0), capital=initial_capital,
                        lookback_window=lookback_window, feature_properties=feature_properties,
                        dynamic_features=dynamic_features, verbose=1)
    policy_kwargs={
        'features_extractor_class': CombinedLSTMExtractor,
        'features_extractor_kwargs': {
            'prices_net_kwargs': {'input_shape': (10, n_assets),
                                  'lstm_hidden_dim': 256,
                                  'lstm_num_layers': 1,
                                  'h_fc_dim': 16,
                                  'c_fc_dim': 32,
                                  'fc_arch': (16, 8)},
            'sequential_net_kwargs': {'input_shape': (10, n_assets),
                                      'lstm_hidden_dim': 256,
                                      'lstm_num_layers': 1,
                                      'h_fc_dim': 16,
                                      'c_fc_dim': 32,
                                      'fc_arch': (32, 16)},
            'fc_arch': (32, 8),
            'device': th.device('cuda')
        }
    }
    model = PPO(policy='MultiInputPolicy', env=env,
                learning_rate=0.01,
                n_steps=256,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.05,
                vf_coef=0.5,
                policy_kwargs=policy_kwargs)

    eval_callback = EvalCallback(Monitor(test_env), eval_freq=256, n_eval_episodes=1, render=False)
    model.learn(total_timesteps=int(1e5), callback=eval_callback)



if __name__ == '__main__':
    np.random.seed(123)
    main()
