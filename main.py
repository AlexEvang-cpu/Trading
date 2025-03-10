import requests
import pandas as pd
import pandas_ta as ta
import time
from flask import Flask, jsonify

# MEXC API Base URL
MEXC_BASE_URL = "https://api.mexc.com/api/v3"

# Flask app for Railway deployment
app = Flask(__name__)

# Function to fetch historical price data
def get_kline_data(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"{MEXC_BASE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad HTTP responses

        data = response.json()
        if not data:
            print("❌ Error: API returned empty data")
            return None

        # Adjusted column names to match MEXC response
        column_names = ["timestamp", "open", "high", "low", "close", "volume", "close_timestamp", "quote_asset_volume"]

        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=column_names)

        # Convert necessary columns to correct types
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

# Function to calculate Bollinger Bands and RSI
def calculate_indicators(df):
    try:
        print("✅ Checking DataFrame before applying indicators:")
        print(df.head())  # Debugging: Show first few rows

        if "close" not in df.columns:
            print("❌ Error: 'close' column missing in DataFrame")
            return None

        # Ensure close price is float
        df["close"] = df["close"].astype(float)

        # Calculate RSI
        if not df["close"].isnull().values.any():
            df["rsi"] = ta.rsi(df["close"], length=14)
        else:
            print("❌ Error: 'close' column contains NaN values")
            return None

        # Calculate Bollinger Bands
        bbands = ta.bbands(df["close"], length=20)
        
        if bbands is None:
            print("❌ Error: Bollinger Bands calculation failed")
            return None

        # Ensure BBANDS keys exist
        expected_keys = ["BBL_20_2.0", "BBU_20_2.0"]
        for key in expected_keys:
            if key not in bbands.columns:
                print(f"❌ Error: Missing column {key} in Bollinger Bands")
                return None

        df["upper_band"] = bbands["BBU_20_2.0"]
        df["lower_band"] = bbands["BBL_20_2.0"]

        print("✅ Successfully calculated indicators")
        return df
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return None

# Function to detect trade signals
def check_trade_signals(df):
    if df is None or df.empty:
        print("❌ Error: DataFrame is empty or None")
        return []
    
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
    
    if df is None:
        return "❌ Error fetching data"

    df = calculate_indicators(df)
    
    if df is None:
        return "❌ Error calculating indicators"

    signals = check_trade_signals(df)
    
    if signals:
        alert_message = f"🚨 Trade Alert 🚨\n" + "\n".join(signals)
        print(alert_message)
        return alert_message
    else:
        return "✅ No trade signals"

@app.route("/")
def run_bot():
    try:
        result = main()
        return jsonify({"status": "running", "message": result if result else "No data available"})
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
