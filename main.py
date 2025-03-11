import os
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify
from datetime import datetime

# Configuration for 1-minute intervals
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")  # 1-minute timeframe
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))       # 5 hours of data

app = Flask(__name__)

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch and process 1-minute market data"""
    endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    
    try:
        response = requests.get(endpoint, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            app.logger.error(f"MEXC API Error: {data.get('message', 'Unknown error')}")
            return None

        raw_data = data.get("data", {})
        if not raw_data.get("time"):
            app.logger.error("Empty data response from API")
            return None

        # Create DataFrame from MEXC's response format
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit="s", utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors="coerce"),
            "high": pd.to_numeric(raw_data["realHigh"], errors="coerce"),
            "low": pd.to_numeric(raw_data["realLow"], errors="coerce"),
            "close": pd.to_numeric(raw_data["realClose"], errors="coerce"),
            "volume": pd.to_numeric(raw_data["vol"], errors="coerce")
        }).dropna()

        if len(df) < 50:  # Require minimum 50 data points
            app.logger.error(f"Insufficient data: {len(df)}/50 candles")
            return None
            
        return df.iloc[-250:]  # Return last ~4 hours of data

    except Exception as e:
        app.logger.error(f"Data fetch error: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators for 1-minute data"""
    try:
        # Core Indicators
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema_20"] = ta.ema(df["close"], length=20)
        
        # Bollinger Bands with volatility
        bbands = ta.bbands(df["close"], length=20, std=2)
        df = pd.concat([df, bbands.add_prefix("bb_")], axis=1)
        
        # Trend and Momentum
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd.add_prefix("macd_")], axis=1)
        
        # Clean and validate
        required_cols = ["close", "rsi", "bb_BBU_20_2.0", "bb_BBL_20_2.0", "ema_20"]
        df = df.dropna().loc[:, required_cols]
        
        return df.iloc[-100:]  # Use most recent 100 periods

    except Exception as e:
        app.logger.error(f"Indicator error: {str(e)}")
        return None

def generate_signals(df):
    """Generate signals optimized for 1-minute trading"""
    try:
        signals = []
        latest = df.iloc[-1]
        
        # Bollinger Band Signals
        if latest["close"] > latest["bb_BBU_20_2.0"]:
            signals.append("Upper Band Breakout")
        elif latest["close"] < latest["bb_BBL_20_2.0"]:
            signals.append("Lower Band Breakdown")
        
        # RSI with EMA Filter
        if latest["rsi"] < 35 and latest["close"] > latest["ema_20"]:
            signals.append("RSI Oversold Bounce")
        elif latest["rsi"] > 65 and latest["close"] < latest["ema_20"]:
            signals.append("RSI Overbought Rejection")

        # Volatility Check
        if latest["atr"] / latest["close"] > 0.003:
            signals.append("High Volatility")
            
        return signals

    except Exception as e:
        app.logger.error(f"Signal error: {str(e)}")
        return []

@app.route("/")
def main_handler():
    try:
        df = get_futures_kline()
        if df is None:
            return jsonify({
                "status": "error",
                "message": "Data unavailable",
                "advice": ["Check symbol/interval", "Verify API status"]
            }), 503
        
        processed_df = calculate_indicators(df)
        if processed_df is None:
            return jsonify({
                "status": "error",
                "message": "Indicator failure",
                "data_points": len(df)
            }), 500

        latest = processed_df.iloc[-1]
        return jsonify({
            "status": "success",
            "signals": generate_signals(processed_df),
            "metrics": {
                "price": round(latest["close"], 2),
                "rsi": round(latest["rsi"], 1),
                "upper_band": round(latest["bb_BBU_20_2.0"], 2),
                "lower_band": round(latest["bb_BBL_20_2.0"], 2),
                "ema_20": round(latest["ema_20"], 2),
                "atr": round(latest["atr"], 2)
            },
            "meta": {
                "symbol": DEFAULT_SYMBOL,
                "interval": DEFAULT_INTERVAL,
                "candles": len(processed_df)
            }
        })

    except Exception as e:
        app.logger.critical(f"System error: {str(e)}")
        return jsonify({"status": "error", "message": "Internal failure"}), 500

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
