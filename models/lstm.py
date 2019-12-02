import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time


class Model:
    
    def __init__(self, ticker, batch_size, epochs, lr, lookback_days, prediction_days, dim):
        self.scaler = MinMaxScaler()
        self.ticker = ticker
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.dim = dim

    def parse_data(self, filename, ticker, dummy=False):
        df = pd.read_csv(filename)
        df = df.drop(df.columns[0],1)
        df = df.loc[df["TICKER"] == ticker]
        if dummy:
            # just take one years worth of data (2017,2018)
            df = df[(df['date'] > '2013-01-01') & (df['date'] < '2018-12-31')] 
        return df

    def trim_dataset(self, data, batch_size):
        n = len(data)
        trim = n % batch_size
        return data[:n-trim]

    def create_flat_price(self, df, feature_columns=['PRC']):
        return df[['PRC']].to_numpy().flatten()
    
    def create_features_labels(self, df, lookback_days, prediction_days, feature_columns=['PRC']):
        # feature columns is the columns to keep as feature for example  feature_columns=['PERMNO', 'date', 'TICKER','PRC', 'VOL']
        # df is the stock price informtion for a single stock
        dim = 1
        dim = df.shape[1]
        X = []
        Y = []
        for i in range(len(df)-lookback_days-prediction_days):
            xi = df[feature_columns][i:i+lookback_days].to_numpy()
            yi = df['PRC'][i+lookback_days:i+lookback_days+prediction_days].to_numpy()
            X.append(xi)
            Y.append(yi)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y
        
    def split_data(self, df, test, cross_validation, batch_size, lookback_days, prediction_days, feature_columns=['PRC']):
        # IMPORTANT NOTE: test is the fraction of the overall dataset (X, y) that will be used for testing
        # so if (X, y) is 1000 records long and test = 0.2 then 800 samples will be used in training and 200 in testing
        # cross_validation is WITHIN the training samples how many will be used for cross validation, so if cross_validation = 0.1
        # then 800 * 0.1 = 80 will be used for cross validation and the other 720 for normal training
        # final splits: N = len(X), X_test = test*N, X_cv = (1-test)*cross_validation*N, X_tr = (1-test)*(1-cross_validation)*N
        X,y = self.create_features_labels(df, lookback_days, prediction_days, feature_columns)
        X_train, X_ts, y_train, y_ts = train_test_split(X, y, test_size=test)
        X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=cross_validation)
        X_tr = self.trim_dataset(X_tr, batch_size)
        X_cv = self.trim_dataset(X_cv, batch_size)
        X_ts = self.trim_dataset(X_ts, batch_size)
        y_tr = self.trim_dataset(y_tr, batch_size)
        y_cv = self.trim_dataset(y_cv, batch_size)
        y_ts = self.trim_dataset(y_ts, batch_size)

        self.X_tr = X_tr; self.X_cv = X_cv; self.X_ts = X_ts; self.y_tr = y_tr; self.y_cv = y_cv; self.y_ts = y_ts
        return X_tr, X_cv, X_ts, y_tr, y_cv, y_ts

    def split_data_last_year_test(self, df, cross_validation, batch_size, lookback_days, prediction_days, feature_columns=['PRC']):
        train_df = df[df['date']<'2018-01-01']
        ts_df = df[df['date']>='2018-01-01']
        # self.scaler.fit(train_df[feature_columns].to_numpy())
        self.train_df = train_df
        self.ts_df = ts_df
        self.flat_train = self.create_flat_price(train_df)
        self.flat_test = self.create_flat_price(ts_df)
        X_train, y_train = self.create_features_labels(train_df, lookback_days, prediction_days, feature_columns)
        # (N, T, D) = X_train.shape
        # X_train = X_train.reshape((N*T, D))
        # X_train = self.scaler.transform(X_train)
        # X_train = X_train.reshape((N, T, D))
        X_ts, y_ts = self.create_features_labels(ts_df, lookback_days, prediction_days, feature_columns)
        X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=cross_validation)
        X_tr = self.trim_dataset(X_tr, batch_size)
        X_cv = self.trim_dataset(X_cv, batch_size)
        X_ts = self.trim_dataset(X_ts, batch_size)
        y_tr = self.trim_dataset(y_tr, batch_size)
        y_cv = self.trim_dataset(y_cv, batch_size)
        y_ts = self.trim_dataset(y_ts, batch_size)
        self.X_tr = X_tr; self.X_cv = X_cv; self.X_ts = X_ts; self.y_tr = y_tr; self.y_cv = y_cv; self.y_ts = y_ts
        return X_tr, X_cv, X_ts, y_tr, y_cv, y_ts
        
    def init_model(self, input_shape):
        # define model
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', input_shape=input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.prediction_days))
        self.model.compile(loss='mse', optimizer='adam')
        return self.model

    def train_model(self, X_tr, X_cv, y_tr, y_cv, save_weights_dir, verbose=2):
        # fit network
        self.model.fit(X_tr, y_tr, validation_data=(X_cv, y_cv), epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
        self.save_model(save_weights_dir)
        return self.model

    def save_model(self, save_weights_dir):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = save_weights_dir+self.ticker + '_'+timestr
        self.model.save(filename)

    def load_model(self, weight_file):
        self.model.load_weights(weight_file)
        # play around with model here if you want
        # testing data is available at self.X_ts and self.y_ts after create_features_labels is run
        # for example
        # pred = model.predict(self.X_t)
        return self.model
        
    
def testing_helper(actual, pred, epochs, ticker, save_figures_dir, save=True):
    actual_0 = actual.flatten()
    pred_0 = pred.flatten()
    plt.figure()
    plt.plot(actual_0, 'g')
    plt.plot(pred_0, 'b')
    plt.title(ticker+" EPOCHS: "+str(epochs))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = save_figures_dir+ticker+'_'+timestr
    if save:
        plt.savefig(filename)

def driver():
    TICKERS = ['TROW', 'CMA', 'BEN', 'WFC', 'JPM', 'BK', 'NTRS', 'AXP', 'BAC', 'USB', 'RJF', 'C', 'STT', 'SCHW', 'COF', 'IVZ', 'ETFC', 'AMG', 'GS', 'BLK', 'AMP', 'DFS']
    EPOCHS = 100
    BATCH_SIZE = 50
    LEARNING_RATE = 0.001
    TEST_RATIO = 0.2
    CROSS_VALIDATION_RATIO = 0.2
    LOOKBACK_DAYS = 30
    PREDICTION_DAYS = 1
    DATA_FILE = "../data/pre_data_10years"
    FEATURE_COLUMNS=['PRC']
    DIM = len(FEATURE_COLUMNS)
    DIRECTORY = '/Users/Sai/Desktop/566/Financial-DL/trained_100_epochs_0001_lr/'
    WEIGHTS_DIR = DIRECTORY+'/weights/'
    FIGURES_DIR = DIRECTORY+'/figures/'
    for TICKER in TICKERS:
        lstm = Model(TICKER, BATCH_SIZE, EPOCHS, LEARNING_RATE, LOOKBACK_DAYS, PREDICTION_DAYS, DIM)
        df = lstm.parse_data(DATA_FILE, TICKER)
        df['PRC'] = np.log(df['PRC'])
        X_tr, X_cv, X_ts, y_tr, y_cv, y_ts = lstm.split_data_last_year_test(df, CROSS_VALIDATION_RATIO, BATCH_SIZE, LOOKBACK_DAYS, PREDICTION_DAYS, FEATURE_COLUMNS) 
        model = lstm.init_model(X_tr.shape[1:])
        model = lstm.train_model(X_tr, X_cv, y_tr, y_cv, WEIGHTS_DIR)
        pred = model.predict(X_ts)
        testing_helper(y_ts, pred, EPOCHS, TICKER, FIGURES_DIR, save=True)

driver()
# EPOCHS = 100
# BATCH_SIZE = 50
# LEARNING_RATE = 0.1
# TEST_RATIO = 0.2
# CROSS_VALIDATION_RATIO = 0.2
# LOOKBACK_DAYS = 30
# PREDICTION_DAYS = 1
# DATA_FILE = "../data/pre_data_10years"
# FEATURE_COLUMNS=['PRC']
# DIM = len(FEATURE_COLUMNS)
# TICKER = 'IVZ'
# lstm = Model(TICKER, BATCH_SIZE, EPOCHS, LEARNING_RATE, LOOKBACK_DAYS, PREDICTION_DAYS, DIM)
# df = lstm.parse_data(DATA_FILE, TICKER)
# df['PRC'] = np.log(df['PRC'])
# X_tr, X_cv, X_ts, y_tr, y_cv, y_ts = lstm.split_data_last_year_test(df, CROSS_VALIDATION_RATIO, BATCH_SIZE, LOOKBACK_DAYS, PREDICTION_DAYS, FEATURE_COLUMNS) 
# model = lstm.init_model(X_tr.shape[1:])
# model = lstm.train_model(X_tr, X_cv, y_tr, y_cv)
# model = lstm.load_model('./saved_weights/BAC_20191201-172602')