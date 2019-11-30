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


def run_model(is_train=True,model_name='rl_model'):
    df = pd.read_csv('./data/db.csv')
    df = df.sort_values('date')
    df = df.drop(columns='date')

    df = df.dropna().reset_index()
    print(df.isnull().sum().sum())

    # train, test = train_test_split(df, test_size=0.1)

    train = df[:int(0.9 * len(df))].reset_index()
    test = df[int(0.9 * len(df)):].reset_index()
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(train, train, 30, True)])
    env_test = DummyVecEnv([lambda: StockTradingEnv(test, train, 30, False)])

    if is_train and model_name == 'rl_rand_model':
        model = PPO2(RandomPolicy, env, verbose=11, tensorboard_log="./log/rand_stock_tensorboard/")

    elif not is_train and model_name == 'rl_rand_model':
        model = PPO2.load("./ckpt/rl_rand_model")

    elif is_train and model_name == 'rl_model':
        model = PPO2(CustomPolicy, env, verbose=11, tensorboard_log="./log/ppo2_stock_tensorboard/")

    elif not is_train and model_name == 'rl_model':
        model = PPO2.load("./ckpt/rl_model")

    elif not is_train and model_name == 'hr_model':
        model = Heristic(env_test)
    elif is_train and model_name == 'hr_model':
        model = Heristic(env)

    elif not is_train and model_name == 'rnn_model':
        model = Baseline(env_test)
    elif is_train and model_name == 'rnn_model':
        model = Baseline(env)
    else:
        assert False

    if not is_train:

        for epoch in range(1):
            obs = env_test.reset()
            rewards = []
            for i in range(len(test.loc[:, 'TROW_PRC'].values) - 30):
                action, _states = model.predict(obs)
                obs, reward, done, info = env_test.step(action)
                rewards.append(reward[0])
                env_test.render()
            plt.plot(rewards)
            plt.show()
    else:

        for epoch in range(1):
            obs = env.reset()
            if model_name =='rl_model':
                model.learn(total_timesteps=500000)
            model.save("./ckpt/"+model_name)
            rewards = []
            for i in range(len(test.loc[:, 'TROW_PRC'].values) - 30):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                rewards.append(reward[0])
                env.render()
            plt.plot(rewards)
            plt.show()



if __name__ == '__main__':
    run_model()
    run_model(False)

