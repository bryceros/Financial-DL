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

class RNN:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def parse_data(self, filename, ticker=None, dummy=False):
        df = pd.read_csv(filename)
        df = df.drop(df.columns[0],1)
        if ticker:
            df = df.loc[df["TICKER"] == ticker]
        if dummy:
            # just take one years worth of data (2017,2018)
            df = df[(df['date'] > '2013-01-01') & (df['date'] < '2018-12-31')] 
        return df

    def trim_dataset(self, data, batch_size):
        n = len(data)
        trim = n % batch_size
        return data[:n-trim]

    def format_data(self, data, batch_size, test_ratio=0.2, lookback_d=90, prediction_d=30 ):
        # note data is already figured by ticker
        lookback_days = lookback_d # number of days we want to base our prediction on
        prediction_days = prediction_d # number of days we want to predict
        
        X = []
        Y = []
        for i in range(len(data)-lookback_days-prediction_days):
            # for debugging purposes this data can be generated with date column
            # xi = data[['date','PRC','VOL']][i:i+lookback_days]
            # yi = data[['date','PRC']][i+lookback_days:i+lookback_days+prediction_days]
            xi = data[['PRC','VOL']][i:i+lookback_days].to_numpy()
            yi = data[['PRC']][i+lookback_days:i+lookback_days+prediction_days].to_numpy()
            X.append(xi)
            Y.append(yi)

        X = np.array(X)
        y = np.array(Y)
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, train_size=(1-test_ratio), test_size=test_ratio,shuffle=False)
        
        N, T, D = X_tr.shape
        X_tr_d2 = X_tr.reshape((N, T*D)) #have to scale a 2d array
        X_tr_d2 = self.scaler.fit_transform(X_tr_d2)
        X_tr = X_tr_d2.reshape((N, T, D))
        
        n, t, d = X_ts.shape
        X_ts_d2 = X_ts.reshape((n, t*d))
        X_ts_d2 = self.scaler.transform(X_ts_d2)
        X_ts = X_ts_d2.reshape((n, t, d))

        X_tr = self.trim_dataset(X_tr, batch_size)
        y_tr = self.trim_dataset(y_tr, batch_size)
        X_ts = self.trim_dataset(X_ts, batch_size)
        y_ts = self.trim_dataset(y_ts, batch_size)
        
        return X_tr, X_ts, y_tr, y_ts

    def init_model(self, batch_size, lookback_days, D, prediction_days, lr):
        self.model = Sequential()
        self.model.add(LSTM(100, batch_input_shape=(batch_size, lookback_days, D), dropout=0.0, recurrent_dropout=0.0, stateful=True,   kernel_initializer='random_uniform'))
        self.model.add(Dense(prediction_days))
        optimizer = tf.optimizers.RMSprop(lr=lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

    def run_model(self, batch_size, epochs, X_tr, X_ts, y_tr, y_ts):
        y_tr = y_tr.reshape((y_tr.shape[0], y_tr.shape[1]))
        y_ts = y_ts.reshape((y_ts.shape[0], y_ts.shape[1]))

        csv_logger = CSVLogger(os.path.join('/Users/Sai/Desktop/566/Financial-DL/runs/', 'sample' + '.log'), append=True)
        history = self.model.fit(X_tr, y_tr, epochs=epochs, verbose=2, batch_size=batch_size, validation_data=(X_ts, y_ts), shuffle=False, callbacks=[csv_logger])

batch_size = 100
lookback_days = 150
prediction_days = 30
dimensions = 2
epochs = 100
rnn = RNN()
df = rnn.parse_data("../data/pre_data_10years", "BAC")
X_tr, X_ts, y_tr, y_ts = rnn.format_data(df, 100, lookback_d=lookback_days, prediction_d=prediction_days)
rnn.init_model(batch_size, lookback_days, dimensions, prediction_days, 0.6)
rnn.run_model(batch_size, epochs, X_tr, X_ts, y_tr, y_ts)