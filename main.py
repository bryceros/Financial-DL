import gym
import json
import datetime as dt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

import matplotlib.pyplot as plt

from model import CustomPolicy
from randmodel import RandomPolicy
from env.StockTradingEnv import StockTradingEnv

import pandas as pd

from utils.utils import plot_learning_curve
from utils.dataparse import parse

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#df = pd.read_csv('./data/AAPL.csv')
#df = df.sort_values('Date')

#df = parse('./data/pre_data_10years')
df = pd.read_csv('./data/db.csv')
df = df.sort_values('date')

df = df.dropna().reset_index() 
print(df.isnull().sum().sum())

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(CustomPolicy, env, verbose=11, tensorboard_log="./log/ppo2_stock_tensorboard/")
#model = PPO2(RandomPolicy, env, verbose=11, tensorboard_log="./log/rand_stock_tensorboard/")
for epoch in range(1):

    model.learn(total_timesteps=200000)

    obs = env.reset()
    rewards,success_rate = [], []
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward[0])
        success_rate.append(info[0]['success_rate'])
        env.render()
    #plot_learning_curve(rewards, success_rate, epoch)
