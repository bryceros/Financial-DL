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
        self.ticker = TICKER
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
        
    def split_data(self, X, y, test, cross_validation, batch_size):
        # IMPORTANT NOTE: test is the fraction of the overall dataset (X, y) that will be used for testing
        # so if (X, y) is 1000 records long and test = 0.2 then 800 samples will be used in training and 200 in testing
        # cross_validation is WITHIN the training samples how many will be used for cross validation, so if cross_validation = 0.1
        # then 800 * 0.1 = 80 will be used for cross validation and the other 720 for normal training
        # final splits: N = len(X), X_test = test*N, X_cv = (1-test)*cross_validation*N, X_tr = (1-test)*(1-cross_validation)*N

        X_train, X_ts, y_train, y_ts = train_test_split(X, y, test_size=test)
        X_tr, X_cv, y_tr, y_cv = train_test_split(X_train, y_train, test_size=cross_validation)
        X_tr = self.trim_dataset(X_tr, batch_size)
        X_cv = self.trim_dataset(X_cv, batch_size)
        X_ts = self.trim_dataset(X_ts, batch_size)
        y_tr = self.trim_dataset(y_tr, batch_size)
        y_cv = self.trim_dataset(y_cv, batch_size)
        y_ts = self.trim_dataset(y_ts, batch_size)
        return X_tr, X_cv, X_ts, y_tr, y_cv, y_ts

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

    def init_model(self):
        self.model = Sequential()
        batch_input_shape = (self.batch_size, self.lookback_days, self.dim)
        self.model.add(LSTM(100, batch_input_shape=batch_input_shape, dropout=0.0, recurrent_dropout=0.0, stateful=True,   kernel_initializer='random_uniform'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(20,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))
        optimizer = tf.optimizers.RMSprop(lr=lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def build_model(self, X_tr, X_cv, y_tr, y_cv, verbose=2):
        # define model
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', input_shape=(X_tr.shape[1], X_tr.shape[2])))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.prediction_days))
        self.model.compile(loss='mse', optimizer='adam')
        # fit network
        self.model.fit(X_tr, y_tr, validation_data=(X_cv, y_cv), epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
        self.save_model()
        
        return self.model

    def save_model(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join('/Users/Sai/Desktop/566/Financial-DL/saved_models',self.ticker + '_'+timestr)
        self.model.save(filename)
        
    # train the model
    def train_model(self, batch_size, epochs, X_tr, X_cv, y_tr, y_cv):
        y_tr = y_tr.reshape((y_tr.shape[0], y_tr.shape[1]))
        y_ts = y_ts.reshape((y_ts.shape[0], y_ts.shape[1]))

        history = self.model.fit(X_tr, y_tr, epochs=epochs, verbose=2, batch_size=batch_size, validation_data=(X_ts, y_ts), shuffle=False)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join('/Users/Sai/Desktop/566/Financial-DL/saved_models/', timestr)+'_weights'
        self.model.save(filename)



BATCH_SIZE = 100
EPOCHS = 2
LEARNING_RATE = 0.6
LOOKBACK_DAYS = 30
PREDICTION_DAYS = 5
TICKER = "BAC"
DATA_FILE = "../data/pre_data_10years"
FEATURE_COLUMNS=['PRC']
DIM = len(FEATURE_COLUMNS)
lstm = Model(TICKER, BATCH_SIZE, EPOCHS, LEARNING_RATE, LOOKBACK_DAYS, PREDICTION_DAYS, DIM)
df = lstm.parse_data(DATA_FILE, TICKER)
X,y = lstm.create_features_labels(df, LOOKBACK_DAYS, PREDICTION_DAYS, feature_columns=['PRC'])
X_tr, X_cv, X_ts, y_tr, y_cv, y_ts = lstm.split_data(X, y, 0.2, 0.2, BATCH_SIZE)
lstm.build_model(X_tr, X_cv, y_tr, y_cv)
