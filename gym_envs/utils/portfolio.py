import numpy as np
import pandas as pd

class Portfolio:
    def __init__(self, initial_capital, lookahead_window=None, max_invest=None, take_profit=None, stop_loss=None):
        self.capital = initial_capital
        self.max_invest = max_invest
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.lookahead_window = lookahead_window
        self.open_positions = pd.DataFrame(
            columns=['asset', 'price', 'amount', 'position', 'take_profit', 'stop_loss', 'time_held'])
        self._idx = 0

    def valuation(self, prices):
        return self.capital + np.sum(
            self.open_positions['asset'].map(prices) * self.open_positions['amount'] * self.open_positions['position'])

    def _trade_positions(self, prices, positions, expected_returns):
        profit_mask = (self.open_positions['asset'].map(prices) -
                       self.open_positions['price']) * self.open_positions['position'] / self.open_positions['price'] >= \
                      self.open_positions['take_profit']
        loss_mask = (self.open_positions['asset'].map(prices) -
                     self.open_positions['price']) * self.open_positions['position'] / self.open_positions['price'] <= \
                    self.open_positions['stop_loss']
        time_mask = self.open_positions['time_held'] >= self.lookahead_window
        closing_positions = self.open_positions.loc[profit_mask | loss_mask | time_mask].copy()
        if len(closing_positions) > 0:
            sale = np.sum(
                closing_positions['amount'] * self.open_positions['asset'].map(prices) * closing_positions['position'])
            self.capital += sale
            self.open_positions = self.open_positions.drop(index=closing_positions.index)
        self.open_positions['time_held'] = self.open_positions['time_held'] + 1
        if isinstance(positions, (int, np.int32, np.int64, float, np.float32, np.float64)):
            positions = np.ones(len(prices))*positions
        if len(positions) > len(prices):
            assert len(positions) == len(prices) + 1, \
                f'One entry in positions may be reserved for holding, but no more. Found {len(prices)} prices, but {len(positions)} positions.'
            positions = positions[1::]

        real_positions = self.max_invest/len(positions) * positions
        for asset, position, r in zip(prices.keys(), real_positions, expected_returns):
            if self.capital > position and position != 0:
                self.open_positions.loc[self._idx] = asset, prices[asset], np.abs(position) / prices[asset], np.sign(
                    position), r, -np.inf, 0
                self.capital -= position
                self._idx += 1