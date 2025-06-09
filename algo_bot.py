
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

std_dev_multiplier = 1.5
short_sma_window = 50
long_sma_window = 200
bb_window = 20
bb_std_dev = 2.0
rsi_window = 14
rsi_band_window = 20
rsi_band_std_dev = 1.0
volume_sma_window = 20

start_date = "2018-01-01"
end_date = "2025-01-01"

gld = yf.download("GLD", start=start_date, end=end_date)
qqq = yf.download("QQQ", start=start_date, end=end_date)

data = pd.DataFrame({
    'GLD': gld['Close'],
    'QQQ': qqq['Close'],
    'Volume': gld['Volume']
})

data.dropna(inplace=True)
required_data_points = max(30, long_sma_window, bb_window, rsi_window, rsi_band_window, volume_sma_window)
if len(data) < required_data_points:
    exit()

data['rolling_corr'] = data['GLD'].rolling(window=30).corr(data['QQQ'])
gld_mean = data['GLD'].mean()
qqq_mean = data['QQQ'].mean()
if qqq_mean == 0:
    exit()

data['spread'] = (data['GLD'] - data['QQQ'] * gld_mean / qqq_mean) / data['QQQ']
spread_mean = data['spread'].rolling(window=30).mean()
spread_std = data['spread'].rolling(window=30).std()
data['mr_buy_signal'] = (data['spread'] < (spread_mean - std_dev_multiplier * spread_std))
data['mr_sell_signal'] = (data['spread'] > (spread_mean + std_dev_multiplier * spread_std))

data['short_sma'] = data['GLD'].rolling(window=short_sma_window).mean()
data['long_sma'] = data['GLD'].rolling(window=long_sma_window).mean()

data['bb_middle_band'] = data['GLD'].rolling(window=bb_window).mean()
data['bb_std_dev'] = data['GLD'].rolling(window=bb_window).std()
data['bb_upper_band'] = data['bb_middle_band'] + (data['bb_std_dev'] * bb_std_dev)
data['bb_lower_band'] = data['bb_middle_band'] - (data['bb_std_dev'] * bb_std_dev)

delta = data['GLD'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
with np.errstate(divide='ignore', invalid='ignore'):
    rs = avg_gain / avg_loss
    rs[np.isinf(rs)] = 100
    rs = rs.fillna(0)
data['rsi'] = 100 - (100 / (1 + rs))

data['rsi_mean_band'] = data['rsi'].rolling(window=rsi_band_window).mean()
data['rsi_std_band'] = data['rsi'].rolling(window=rsi_band_window).std()
data['rsi_upper_band'] = data['rsi_mean_band'] + (data['rsi_std_band'] * rsi_band_std_dev)
data['rsi_lower_band'] = data['rsi_mean_band'] - (data['rsi_std_band'] * rsi_band_std_dev)
data['rsi_buy_signal'] = (data['rsi'] < data['rsi_lower_band'])
data['rsi_sell_signal'] = (data['rsi'] > data['rsi_upper_band'])

data['volume_sma'] = data['Volume'].rolling(window=volume_sma_window).mean()

data['position'] = 0
for i in range(1, len(data)):
    prev_position = data['position'].iloc[i-1]
    if (pd.isna(data['mr_buy_signal'].iloc[i]) or pd.isna(data['mr_sell_signal'].iloc[i]) or
        pd.isna(data['short_sma'].iloc[i]) or pd.isna(data['long_sma'].iloc[i]) or
        pd.isna(data['bb_lower_band'].iloc[i]) or pd.isna(data['bb_upper_band'].iloc[i]) or
        pd.isna(data['rsi_lower_band'].iloc[i]) or pd.isna(data['rsi_upper_band'].iloc[i]) or
        pd.isna(data['volume_sma'].iloc[i]) or pd.isna(data['Volume'].iloc[i])):
        data.at[data.index[i], 'position'] = prev_position
        continue

    buy_condition = (
        data['mr_buy_signal'].iloc[i] and
        data['short_sma'].iloc[i] > data['long_sma'].iloc[i] and
        data['GLD'].iloc[i] <= data['bb_lower_band'].iloc[i] and
        data['rsi_buy_signal'].iloc[i] and
        data['Volume'].iloc[i] > data['volume_sma'].iloc[i] and
        prev_position == 0
    )
    sell_condition = (
        data['mr_sell_signal'].iloc[i] and
        data['short_sma'].iloc[i] < data['long_sma'].iloc[i] and
        data['GLD'].iloc[i] >= data['bb_upper_band'].iloc[i] and
        data['rsi_sell_signal'].iloc[i] and
        data['Volume'].iloc[i] > data['volume_sma'].iloc[i] and
        prev_position == 1
    )

    if buy_condition:
        data.at[data.index[i], 'position'] = 1
    elif sell_condition:
        data.at[data.index[i], 'position'] = 0
    else:
        data.at[data.index[i], 'position'] = prev_position

data['position'] = data['position'].fillna(0)
data['GLD_returns'] = data['GLD'].pct_change()
data['strategy_returns'] = data['GLD_returns'] * data['position'].shift(1).fillna(0)
data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod().fillna(1)
data['cumulative_GLD_returns'] = (1 + data['GLD_returns']).cumprod().fillna(1)

plt.figure(figsize=(16, 22))

plt.subplot(7,1,1)
plt.plot(data.index, data['GLD'], label='GLD Price', color='gold')
buy_indices = data.index[(data['position'].diff() == 1)]
sell_indices = data.index[(data['position'].diff() == -1)]
plt.scatter(buy_indices, data['GLD'].loc[buy_indices], label='Buy', marker='^', color='green', s=100)
plt.scatter(sell_indices, data['GLD'].loc[sell_indices], label='Sell', marker='v', color='red', s=100)
plt.legend(); plt.grid(True)

plt.subplot(7,1,2)
plt.plot(data.index, data['spread'], label='Spread', color='purple')
plt.plot(data.index, spread_mean, label='Mean', color='gray', linestyle='--')
plt.plot(data.index, spread_mean - std_dev_multiplier * spread_std, label='Lower Bound', color='red', linestyle=':')
plt.plot(data.index, spread_mean + std_dev_multiplier * spread_std, label='Upper Bound', color='green', linestyle=':')
plt.legend(); plt.grid(True)

plt.subplot(7,1,3)
plt.plot(data.index, data['GLD'], label='GLD Price', color='gold')
plt.plot(data.index, data['short_sma'], label='Short SMA', color='blue')
plt.plot(data.index, data['long_sma'], label='Long SMA', color='red')
plt.legend(); plt.grid(True)

plt.subplot(7,1,4)
plt.plot(data.index, data['GLD'], label='GLD Price', color='gold')
plt.plot(data.index, data['bb_middle_band'], label='Middle Band', color='gray')
plt.plot(data.index, data['bb_upper_band'], label='Upper Band', color='green')
plt.plot(data.index, data['bb_lower_band'], label='Lower Band', color='red')
plt.fill_between(data.index, data['bb_lower_band'], data['bb_upper_band'], color='lightgray', alpha=0.2)
plt.legend(); plt.grid(True)

plt.subplot(7,1,5)
plt.plot(data.index, data['rsi'], label='RSI', color='purple')
plt.plot(data.index, data['rsi_upper_band'], label='Upper Band', color='red')
plt.plot(data.index, data['rsi_lower_band'], label='Lower Band', color='green')
plt.fill_between(data.index, data['rsi_lower_band'], data['rsi_upper_band'], color='lightgray', alpha=0.2)
plt.legend(); plt.grid(True)

plt.subplot(7,1,6)
plt.bar(data.index, data['Volume'], label='Volume', color='lightblue')
plt.plot(data.index, data['volume_sma'], label='Volume SMA', color='darkblue')
plt.legend(); plt.grid(True)

plt.subplot(7,1,7)
plt.plot(data.index, data['cumulative_strategy_returns'], label='Strategy', color='blue')
plt.plot(data.index, data['cumulative_GLD_returns'], label='Buy & Hold', color='orange')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

total_strategy_return = data['cumulative_strategy_returns'].iloc[-1] - 1 if not data['cumulative_strategy_returns'].empty else 0
total_hold_return = data['cumulative_GLD_returns'].iloc[-1] - 1 if not data['cumulative_GLD_returns'].empty else 0

print(f"{total_strategy_return:.2%}")
print(f"{total_hold_return:.2%}")

