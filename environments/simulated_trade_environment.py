import math
import random
import sys
import time

import gym
import numpy as np
from gym import spaces

from config import (INSTRUMENTS, INTERVALS, MAX_STEP_PERCENT, TRADE_INTERVAL, BACKTEST_RANGE,
                    TRAIN_DATA_RANGE, TRANSACTION_FEE, WINDOW, LEVERAGE)
from data.data_manager_multi_processing import DataManager
from environments.portfolio import Portfolio
from utilities.functions import str2time, time2str

MAX_REWARD = sys.maxsize
np.set_printoptions(suppress=True)


class SimulatedTradingEnvironment(gym.Env):
    """ Backtesting trading environment for OpenAI gym """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 num_of_instruments: int = len(INSTRUMENTS),
                 window_size: int = WINDOW,
                 transaction_fee: float = TRANSACTION_FEE,
                 trade_interval: int = TRADE_INTERVAL,
                 max_train_step_percent: int = MAX_STEP_PERCENT,
                 backtest: bool = False,
                 debug: bool = False,
                 log: int = 10):

        super(SimulatedTradingEnvironment, self).__init__()

        # Basic parameters
        self.env = 'Exchange'
        self.n = num_of_instruments
        self.fee = transaction_fee
        self.window_size = window_size
        self.debug = debug
        self.portfolio = Portfolio(self.n, self.fee)
        self.data_manager = DataManager(training=(not backtest))
        self.data_manager.start()
        self.data_manager.waitToStart()
        self.backtest = backtest
        if backtest:
            data_range = BACKTEST_RANGE
        else:
            data_range = TRAIN_DATA_RANGE
        self.log = log
        self.log_counter = 0

        # Backtest parameters
        self.trade_interval = trade_interval  # in sec
        start, end = data_range
        start = start - start % trade_interval
        end = end - end % trade_interval - self.trade_interval
        self.backtest_range = (start, end)  # in sec
        self.step_range = (0, (end - start)/trade_interval)
        self.total_step = (end - start)/trade_interval
        self.current_step = random.randint(0, self.step_range[1])
        self._max_train_step_percent = max_train_step_percent
        temp = self.current_step + self.total_step * self._max_train_step_percent
        self.max_step = temp if temp <= self.step_range[1] else self.step_range[1]

        # Environment parameters
        self.reward_range = (0, self._reward_function(
            MAX_REWARD, delay_factor=False))
        self.observation_space = spaces.Box(low=np.zeros([num_of_instruments, len(INTERVALS), window_size, 7]),
            high=np.ones([num_of_instruments, len(INTERVALS), window_size, 7]), dtype=np.float16)
        self.action_space = spaces.Box(
            low=np.zeros([self.n+1]), high=np.ones([self.n+1]), dtype=np.float16)

        # Private variables
        previous_shares = [0.0 for i in range(self.n)]
        previous_shares.append(1.0)
        self._previous_shares = np.array(previous_shares)
        self._previous_portfolio = np.array(previous_shares)
        self._previous_value = 1.0
        self._start_step = self.current_step
        self._action = None

    def _current_time(self):
        init_time, _ = self.backtest_range
        return init_time + self.current_step * self.trade_interval

    def _current_price_in_trades(self, include_usd=True):
        time_range = self.trade_interval
        timestamp = self._current_time()
        prices = []
        failed = 0

        # Get prices
        for pair in self.portfolio.pairs:
            price_array = self.data_manager.getTradesInRange(
                (timestamp, timestamp+time_range), pair, True)
            if len(price_array) <= 0:
                prices.append(self.data_manager.getOHCL(
                    timestamp+time_range, pair=pair, window=1, numpy_array=True)[0][-1][5])
                failed += 1
            else:
                prices.append(np.average(price_array))

        # Return failed trades
        if failed > self.n/2:
            return False, np.array([])

        # Append usd to the end
        if include_usd:
            prices.append(1.0)

        return True, np.array(prices)

    def _current_price(self, include_usd=True):
        timestamp = self._current_time() + self.trade_interval
        if self.debug:
            dataframe = np.random.rand(5, 3, 16, 7) + 10
        else:
            dataframe = self.data_manager.getDataframe(
                timestamp, normalize=False)
        prices = dataframe[:, 0, -1, 5]
        disturb = (np.random.rand(self.n) - 0.5)/1000. + 1.
        prices = prices * disturb
        if include_usd:
            prices = np.append(prices, 1.0)
        return True, np.array(prices)

    def _reward_function(self, reward, delay_factor=True):
        factor = 1.
        if reward <= 0:
            reward = 0.
        if delay_factor:
            factor = 1./(self.total_step * self._max_train_step_percent -
                         (self.current_step - self._start_step))
        return factor * math.log10(reward+1.0) * 1e3

    def _next_observation(self):
        timestamp = self._current_time()
        if not self.debug:
            return self.data_manager.getDataframe(timestamp, normalize=True)
        else:
            return np.random.rand(5, 3, 16, 7) - 0.5

    def _take_action(self, action):
        previous_portfolio = self._previous_portfolio
        success, current_prices = self._current_price(include_usd=True)
        self._previous_portfolio = action
        if success:
            # Calculate new value
            current_portfolio = action
            self.portfolio.updatePorfolio(action)
            value = (current_prices * self._previous_shares).sum()

            # Apply fee
            portion = np.sum(
                np.abs(previous_portfolio[:-1] - current_portfolio[:-1]))
            # print('Portion', portion,previous_portfolio,current_portfolio)
            self.portfolio.applyFee(portion)

            # Update net worth
            self.portfolio.updateNetWorth(
                value/self._previous_value)
            self._previous_shares = value * current_portfolio / current_prices
            self._previous_value = value
        else:
            # Failed to trade
            timestamp = self._current_time()
            print('Simulated Exchange: Failed to trade between %d and %d' %
                  (timestamp, timestamp+self.trade_interval))

    def step(self, action):
        # action = (action + 1.0)/2.0
        if np.sum(action) != 0:
            action = action/np.sum(action)
        else:
            action = np.zeros(self.n+1)
            action[self.n] = 1.0

        self._action = action
        done = False
        if self.current_step >= self.max_step or self.portfolio.netWorth() <= 1.-1./LEVERAGE:
            # Finishied or Bankrupt
            done = True
            reward = self._reward_function(self.portfolio.netWorth())

            if self.portfolio.netWorth() <= 1.-1./LEVERAGE:
                print('Simulated Exchange: Bankrupt, Reward -1')
                reward = -1.
            else:
                print('Simulated Exchange: Max step reached')
            obs = self.reset()
        else:
            # Update step
            self.current_step += 1
            obs = self._next_observation()
            self._take_action(action)
            reward = self._reward_function(self.portfolio.netWorth())

        if self.log <= self.log_counter:
            self.render()
            self.log_counter = 0
        else:
            self.log_counter += 1
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        # TODO: Unfinished log process
        print()
        print('Step:', self.current_step)
        reward = self._reward_function(self.portfolio.netWorth())
        print('Reward:', reward)
        print('Time:', time2str(self._current_time()))
        print('Portfolio:', self._action)
        print('Prices:', self._current_price()[1][:-1])
        print('Net Worth:', self.portfolio.netWorth(1000))
        pass

    def reset(self):
        # Reset portfolio
        self.portfolio = Portfolio(self.n, self.fee)

        # Reset steps
        self.current_step = random.randint(0, self.step_range[1])
        temp = self.current_step + \
            self.total_step * self._max_train_step_percent
        self.max_step = temp if temp <= self.step_range[1] else self.step_range[1]
        self._start_step = self.current_step

        # Reset temporary variables
        previous_shares = [0.0 for i in range(self.n)]
        previous_shares.append(1.0)
        self._previous_shares = np.array(previous_shares)
        self._previous_portfolio = np.array(previous_shares)
        self._previous_value = 1.0

        return self._next_observation()


if __name__ == '__main__':
    env = SimulatedTradingEnvironment(debug=False, backtest=False)
    while(True):
        # action = np.random.rand(6)
        # action = action / np.sum(action)
        action = np.array([0., 0., 0., 0., 0., -1.])
        env.step(action)
        # env.render()
        # time.sleep(0.1)
