import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify

# MEXC Futures API Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"

app = Flask(__name__)

def get_futures_kline(symbol="BTC_USDT", interval="Min1", limit=100):
    """Fetch futures kline data from MEXC"""
    endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
    params = {
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("data"):
            print(f"❌ API Error: {data.get('message', 'No data returned')}")
            return None

        # Process kline data (columns confirmed from MEXC documentation)
        columns = [
            "timestamp", "open", "high", "low", "close",
            "volume", "amount", "trade_count"
        ]
        
        df = pd.DataFrame(data["data"], columns=columns)
        
        # Convert types and handle timestamps (in milliseconds)
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        return df

    except Exception as e:
        print(f"❌ Data fetch error: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return None
    
    try:
        # Calculate RSI
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        # Calculate Bollinger Bands
        bbands = ta.bbands(df["close"], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)
        
        # Clean up column names
        df.rename(columns={
            "BBU_20_2.0": "upper_band",
            "BBL_20_2.0": "lower_band"
        }, inplace=True)
        
        return df.dropna()
    
    except Exception as e:
        print(f"❌ Indicator error: {str(e)}")
        return None

def generate_signals(df):
    """Generate trading signals"""
    if df is None or df.empty:
        return []
    
    latest = df.iloc[-1]
    signals = []
    
    # RSI Signals
    if latest["rsi"] > 70:
        signals.append("RSI Overbought (Sell)")
    elif latest["rsi"] < 30:
        signals.append("RSI Oversold (Buy)")
    
    # Bollinger Bands Signals
    if latest["close"] > latest["upper_band"]:
        signals.append("Price Above Upper Band (Potential Short)")
    elif latest["close"] < latest["lower_band"]:
        signals.append("Price Below Lower Band (Potential Long)")
    
    return signals

@app.route("/")
def main():
    try:
        # Fetch and process data
        df = get_futures_kline()
        if df is None:
            return jsonify({"error": "Failed to fetch data"}), 500
        
        df = calculate_indicators(df)
        if df is None:
            return jsonify({"error": "Failed to calculate indicators"}), 500
        
        # Generate and return signals
        signals = generate_signals(df)
        return jsonify({
            "status": "success",
            "signals": signals,
            "latest_data": {
                "price": df.iloc[-1]["close"],
                "rsi": round(df.iloc[-1]["rsi"], 2),
                "upper_band": df.iloc[-1]["upper_band"],
                "lower_band": df.iloc[-1]["lower_band"]
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
