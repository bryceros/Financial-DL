import gym
import json
import datetime as dt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

from matplotlib import pyplot as plt

from model import CustomPolicy
from randmodel import RandomPolicy
from env.StockTradingEnv import StockTradingEnv
from baselinemodel import Baseline
from heristicmodel import Heristic

import pandas as pd

from utils.utils import plot_learning_curve, AverageMeter
from utils.dataparse import parse

from sklearn.model_selection import train_test_split
import numpy as np



#df = pd.read_csv('./data/AAPL.csv')
#df = df.sort_values('Date')

#df = parse('./data/pre_data_10years')

df = pd.read_csv('./data/db.csv')
df = df.sort_values('date')
df = df.drop(columns='date')

df = df.dropna().reset_index() 
print(df.isnull().sum().sum())

#train, test = train_test_split(df, test_size=0.1)

train = df[:int(0.9*len(df))].reset_index() 
test = df[int(0.9*len(df)):].reset_index() 
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(train,train,30,True)])
env_test = DummyVecEnv([lambda: StockTradingEnv(test,train,30,False)])

#model = PPO2(CustomPolicy, env, verbose=11, tensorboard_log="./log/ppo2_stock_tensorboard/")
#model = PPO2(RandomPolicy, env, verbose=11, tensorboard_log="./log/rand_stock_tensorboard/")
#model = Baseline(env)
#model = Heristic(env)
for epoch in range(1):

    obs = env_test.reset()
    #model.load("./ckpt/rl_model")
    #model.learn(total_timesteps=500000)
    #model.save("./ckpt/rl_model")
    rewards = []
    for i in range(len(test.loc[:, 'TROW_PRC'].values)-30):
        print(i)
        action, _states = model.predict(obs)
        obs, reward, done, info = env_test.step(action)
        rewards.append(reward[0])
        env_test.render()
    #plt.plot(data=rewards)
    #plt.show()
