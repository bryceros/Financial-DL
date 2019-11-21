import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 200000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, comp=['TROW', 'CMA', 'BEN', 'WFC', 'JPM', 'BK', 'NTRS', 'AXP', 'BAC', 'USB', 'MS', 'RJF', 'C', 'STT', 'SCHW', 'COF', 'IVZ','ETFC','AMG','GS','BLK','AMP','DFS'],attr = ['PRC','VOL','BID','ASK']):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.comp = comp
        self.attr = attr

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        '''self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)'''
        
        '''self.action_space = spaces.Tuple(tuple([spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)]*len(comp)))'''
        
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([len(comp),3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        '''self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,6), dtype=np.float16)'''
        '''self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(comp)*len(attr)+1,6), dtype=np.float16)'''
        self.observation_space = spaces.Box(low=0, high=1, shape=(6+1,max(len(comp)*len(attr),4*len(comp)+2)), dtype=np.float16)
    
    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        '''frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])'''

        #self.current_step = 396
        frame = []
        for c in self.comp:
            for a in self.attr:
                if a == 'VOL':
                    frame.append(self.df.loc[self.current_step: self.current_step +
                            5, c+"_"+a].values / MAX_NUM_SHARES)
                else:
                    frame.append(self.df.loc[self.current_step: self.current_step +
                            5, c+"_"+a].values / MAX_SHARE_PRICE)
        frame = np.array(frame).T

        #print('frame:',frame.shape)

        # Append additional data and scale each value to between 0-1
        profile = np.concatenate([
            np.array([self.balance / MAX_ACCOUNT_BALANCE]),
            np.array([self.net_worth / MAX_ACCOUNT_BALANCE]),
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]).ravel()
        
        frame_pad = np.zeros((frame.shape[0],max(profile.shape[0],frame.shape[1])))
        frame_pad[:frame.shape[0],:frame.shape[1]] = frame
        
        profile_pad = np.zeros((max(profile.shape[0],frame.shape[1])))
        profile_pad[:profile.shape[0]] = profile
        profile_pad = [profile_pad]
        
        obs = np.concatenate((frame_pad,profile_pad))
        #if(obs.shape != (len(self.comp)*len(self.attr)+1,6) ): print('error self.current_step:',self.current_step,", obs.shape:",obs.shape)
        return obs
    
    def _take_action(self, action):
        # Set the current price to a random price within the time step
        '''current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])'''
        i = np.int(np.floor(action[0]))
        #i = 0
        current_price = self.df.loc[self.current_step,self.comp[i]+'_PRC']
        total_price = np.array([self.df.loc[self.current_step,c+'_PRC'] for c in self.comp])
        action_type = action[-2]
        amount = action[-1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis[i] * self.shares_held[i]
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis[i] = (
                prev_cost + additional_cost) / (self.shares_held[i] + shares_bought)
            self.shares_held[i] += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held[i] * amount)
            self.balance += shares_sold * current_price
            self.shares_held[i] -= shares_sold
            self.total_shares_sold[i] += shares_sold
            self.total_sales_value[i] += shares_sold * current_price

        self.net_worth = self.balance + np.sum(self.shares_held * total_price)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held[i] == 0:
            self.cost_basis[i] = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'TROW_PRC'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.net_worth * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()
        info = {
            'success_rate':((self.net_worth - INITIAL_ACCOUNT_BALANCE)/MAX_ACCOUNT_BALANCE)
        }
        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        '''self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0'''
        self.shares_held = np.zeros(len(self.comp))
        self.cost_basis = np.zeros(len(self.comp))
        self.total_shares_sold = np.zeros(len(self.comp))
        self.total_sales_value = np.zeros(len(self.comp))

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'TROW_PRC'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
