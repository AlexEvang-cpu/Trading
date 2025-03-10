import requests
import pandas as pd
import pandas_ta as ta
import time
from flask import Flask

# MEXC API Base URL
MEXC_BASE_URL = "https://api.mexc.com/api/v3"

# Flask app for Railway deployment
app = Flask(__name__)

# Function to fetch historical price data
def get_kline_data(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"{MEXC_BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
        df["close"] = df["close"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    return None

# Function to calculate Bollinger Bands and RSI
def calculate_indicators(df):
    df["rsi"] = df["close"].ta.rsi(length=14)
    bbands = df["close"].ta.bbands(length=20)
    df["upper_band"] = bbands["BBU_20_2.0"]
    df["lower_band"] = bbands["BBL_20_2.0"]
    return df

# Function to detect trade signals
def check_trade_signals(df):
    latest = df.iloc[-1]
    signals = []
    if latest["rsi"] > 70:
        signals.append("RSI Overbought - Potential Sell Signal")
    elif latest["rsi"] < 30:
        signals.append("RSI Oversold - Potential Buy Signal")
    if latest["close"] >= latest["upper_band"]:
        signals.append("Price hit upper Bollinger Band - Possible Reversal Down")
    elif latest["close"] <= latest["lower_band"]:
        signals.append("Price hit lower Bollinger Band - Possible Reversal Up")
    return signals

# Main function to run the bot
def main():
    df = get_kline_data()
    if df is not None:
        df = calculate_indicators(df)
        signals = check_trade_signals(df)
        if signals:
            print(f"🚨 Trade Alert 🚨\n" + "\n".join(signals))

@app.route("/")
def run_bot():
    main()
    return "Crypto bot is running!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
