import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn as sk
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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

    def format_data(self, data, test_ratio=0.2, lookback_d=90, prediction_d=30 ):
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

        return X_tr, X_ts, y_tr, y_ts


rnn = RNN()
df = rnn.parse_data("../data/pre_data_10years", "BAC", dummy=True)
X_tr, X_ts, y_tr, y_ts = rnn.format_data(df)