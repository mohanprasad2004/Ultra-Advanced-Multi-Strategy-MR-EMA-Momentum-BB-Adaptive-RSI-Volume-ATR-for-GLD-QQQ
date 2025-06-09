📈 Multi-Strategy GLD Trading Bot
This Python project implements a multi-strategy trading system for the GLD ETF (Gold) using a combination of technical indicators and strategies. The system is designed for backtesting purposes, comparing the strategy's cumulative returns with a simple Buy & Hold approach.

⚙️ Strategies Used
Mean Reversion (GLD vs QQQ Spread)

Momentum (SMA Crossover)

Bollinger Bands

Adaptive RSI (Relative Strength Index with Dynamic Bands)

Volume Confirmation

Trades are only executed when all strategy conditions align, aiming for high-conviction entry and exit signals.

📊 Data
Source: Yahoo Finance via yfinance

Instruments:

GLD – SPDR Gold Shares ETF (target asset)

QQQ – Invesco Nasdaq 100 ETF (used for mean-reversion spread calculation)

Time Period: 2018-01-01 to 2025-01-01

🔍 Dependencies
bash
Copy
Edit
pip install yfinance pandas numpy matplotlib
🧠 Key Concepts
Mean Reversion: Compares normalized price spread between GLD and QQQ.

Momentum: 50/200-day Simple Moving Average crossover.

Bollinger Bands: Detects price extremes using standard deviation bands.

Adaptive RSI: RSI with dynamic bands for overbought/oversold detection.

Volume Filter: Ensures signals are backed by increased market activity.

📈 Output
The script generates:

Buy/Sell signal plots

Indicator visualizations

Cumulative returns comparison between the strategy and buy-and-hold
