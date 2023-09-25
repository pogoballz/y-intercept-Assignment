import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation, GRU, ReLU, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import pickle

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import itertools
import datetime 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# pyfolio python library for performance and risk analysis
import pyfolio as pf

# load original csv file
raw_data = pd.read_csv("data.csv")


# deserialize cleaned mapping
with open('cleanedData.pickle', 'rb') as handle:
    df = pickle.load(handle)


# risk-free rate for calculating Sharpe
rfr = 0.02

# calculate maximum drawdown
def calculate_max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()
    return max_drawdown

# Calculate Sortino ratio
def calculate_sortino_ratio(returns, risk_free_rate=0):
    avg_return = np.mean(returns - risk_free_rate)
    downside_std = np.std(np.minimum(returns - risk_free_rate, 0))
    sortino_ratio = avg_return / downside_std
    return sortino_ratio