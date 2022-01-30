import pandas as pd
import sys
import numpy as np
import random
from collections import deque
from sklearn import preprocessing
import pickle

import pandas_ta as ta


SEQ_LEN = 60
FUTURE_PREDICT_PERIOD = 1
DATA_PATH = "Binance_BTCUSDT_1h.csv"

pd.set_option('use_inf_as_na', True)

def print_df(df):
    for i in range(0, len(df)):
        print(df.loc[i])

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def preprocess_df(df):
    df.drop(['date', 'symbol', 'volume_usdt','tradecount','future'], axis=1, inplace=True)

    df = df.copy()
    for col in df.columns:
        if "RSI" in col:
            #print(df[col].values)

            df.loc[df[col] >= 70, col] = 1
            df.loc[df[col] <= 30, col] = -1 
            df.loc[(df[col] > 30) & (df[col] < 70), col] = 0  
            

            #print(df[col].values)
            #print(df.head())
        elif col != "target":

            df[col] = df[col].pct_change()
            
            clean_dataset(df)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
        
    print(df.head())
    seq_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            seq_data.append([np.array(prev_days), i[-1]])

    np.random.shuffle(seq_data)

    buys = []
    sells = []

    for seq, target in seq_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    np.random.shuffle(sells)
    np.random.shuffle(buys)

    low = min(len(sells), len(buys))

    buys = buys[:low]
    sells = sells[:low]

    seq_data = buys + sells
    np.random.shuffle(seq_data)

    X = []
    y = []

    for seq, target in seq_data:
        X.append(seq)
        y.append(target)

    #print(df.head())
    return np.array(X), np.array(y)

def classify(cur, fut):
    if float(fut) > float(cur):
        return 1
    else:
        return 0


loaded = True 
try:
    with open(f'{DATA_PATH}_saved_data.pkl', 'rb') as f:
        print("Data found, loading from file.")
        train_X, train_y, val_X, val_y = pickle.load(f)
except:
    print("No existing file found, generating data.")
    loaded = False 


if loaded == False:
    df = pd.read_csv(DATA_PATH, names=["unix", "date", "symbol", "open", "high", "low", "close", "volume_btc", "volume_usdt", "tradecount"], sep=",") 
    
    df = df.loc[::-1]
    df.reset_index(drop=True, inplace=True)
    
    print(df.tail())

    df.ta.rsi(cumulative=True, append=True)
    df.ta.sma(cumulative=True, append=True)
    df.ta.bbands(cumulative=True, append=True)
    
    df.ta.macd(cumulative=True, append=True)
    df.set_index('unix', inplace=True)
    #df.ta.obv(cumulative=True, append=True)
    
    #print(df.columns)
    #print(len(df.columns))
    # df2 = df.iloc[[0, -1]]
    #print(df2.head())

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    
    #print(df.columns)
    df['future'] = df['close'].shift(-FUTURE_PREDICT_PERIOD)
    df['target'] = list(map(classify, df['open'], df['future']))

    times = sorted(df.index.values, reverse=False)
    last_5pct = times[-int(0.05*len(times))]

    #print(f"Times, 5pct_len = {last_5pct}")
    val_df = df[(df.index >= last_5pct)]
    train_df = df[(df.index < last_5pct)]
    print(f"val_df len = {len(val_df)}, train_df len = {len(train_df)}")

    train_X, train_y = preprocess_df(train_df)
    val_X, val_y = preprocess_df(val_df) 
    print(f"val_X len = {len(val_X)}")

    with open(f'{DATA_PATH}_saved_data.pkl', 'wb') as f:
        pickle.dump([train_X, train_y, val_X, val_y], f)
    
    val_df.to_pickle('val_df.pkl')

