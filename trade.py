import numpy as np
import pandas as pd
#import data
import pickle

DATA_PATH = "Binance_BTCUSDT_1h.csv"

STARTING_BAL = 100

# with open(f'{DATA_PATH}_saved_data.pkl', 'rb') as f:
    # print("Data found, loading from file.")
    # train_X, train_y, val_X, val_y = pickle.load(f)
with open("prediction.pkl", 'rb') as f:
    predictions = pickle.load(f)
    print("Predictions loaded successfully!")

df = pd.read_csv(DATA_PATH, names=["unix", "date", "symbol", "open", "high", "low", "close", "volume_btc", "volume_usdt", "tradecount"], sep=",") 

df = df.loc[::-1]
df.reset_index(drop=True, inplace=True)

df.drop('unix', axis=1, inplace=True)
df.dropna(inplace=True)

#val_df = pd.read_pickle('val_df.pkl')
times = sorted(df.index.values, reverse=False)
# last_5pct = times[-int(0.05*len(times))]
last_5pct = times[-len(predictions)-1]
val_df = df[(df.index >= last_5pct)]
val_df.reset_index(drop=True, inplace=True)

print(val_df.head())
print(val_df.tail())
#print(f"val_X length = {len(val_X)}")


price_bought = 0
price_sold = 0
holdings = 1
pct = 0
i = 0 
print(f"Length of predictions = {len(predictions)}")
for pre in predictions:
    if price_bought == 0:
        price_bought = val_df.loc[i].open
        first_price = val_df.loc[i].open
        i += 1

    signal = np.argmax(pre) 
    #print(val_df.loc[i])

    
    # sell
    if holdings != 0 and signal == 0:
        diff_pct = (val_df.loc[i].close - price_bought)/100
        pct += diff_pct 
        holdings = 0

        print(f"Sold with pct change of {diff_pct}%")
    # buy
    elif signal == 1:
        price_bought = val_df.loc[i].open
        holdings = 1

    i += 1
    

i -= 1
price_pct = (val_df.iloc[i].close - first_price)/100

print(f"Total pct change of {pct}% over {i} trades")
print(f"Total price change over time {price_pct}%")
