import os
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify
from datetime import datetime
from joblib import load

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))  # 5 hours of 1m data

# ML Configuration
MODEL_PATH = "model.joblib"
app = Flask(__name__)

# Load ML model
try:
    model = load(MODEL_PATH)
except:
    model = None

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch 1-minute market data"""
    endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            return None

        raw_data = data.get("data", {})
        if not raw_data or "time" not in raw_data:
            return None

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit="s", utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors="coerce"),
            "high": pd.to_numeric(raw_data["realHigh"], errors="coerce"),
            "low": pd.to_numeric(raw_data["realLow"], errors="coerce"),
            "close": pd.to_numeric(raw_data["realClose"], errors="coerce"),
            "volume": pd.to_numeric(raw_data["vol"], errors="coerce")
        }).dropna()

        return df.iloc[-250:]  # Last 4+ hours of data

    except Exception as e:
        app.logger.error(f"Data error: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical features"""
    try:
        # Core Indicators
        df["rsi"] = ta.rsi(df["close"], length=14)
        bbands = ta.bbands(df["close"], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)
        
        # Volatility Measures
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["volatility"] = df["close"].rolling(20).std()
        
        # Momentum Indicators
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # Pattern Recognition
        df["bullish_engulfing"] = ta.cdl_pattern(df["open"], df["high"], df["low"], df["close"], name="engulfing")
        
        # ML Features
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["volume_change"] = df["volume"].pct_change()
        
        return df.dropna().iloc[-100:]

    except Exception as e:
        app.logger.error(f"Indicator error: {str(e)}")
        return None

def generate_ml_signal(df):
    """Generate machine learning prediction"""
    if model is None or len(df) < 50:
        return None
    
    try:
        # Prepare features
        features = df[["rsi", "close", "volatility", "MACD_12_26_9", "volume_change"]].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(features)
        return "Bullish" if prediction[0] == 1 else "Bearish"
    except Exception as e:
        app.logger.error(f"ML error: {str(e)}")
        return None

def generate_signals(df):
    """Generate trading signals"""
    try:
        signals = []
        latest = df.iloc[-1]
        
        # Volatility Entry Condition
        volatility_ratio = latest["atr"] / latest["close"]
        if volatility_ratio > 0.005:
            signals.append("High Volatility Entry")
        
        # Bollinger Band Strategy
        if latest["close"] > latest["BBU_20_2.0"]:
            signals.append("Upper Band Breakout")
        elif latest["close"] < latest["BBL_20_2.0"]:
            signals.append("Lower Band Breakdown")
        
        # RSI Strategy
        if latest["rsi"] < 35 and latest["close"] > latest["ema_20"]:
            signals.append("RSI Oversold Bounce")
        elif latest["rsi"] > 65 and latest["close"] < latest["ema_20"]:
            signals.append("RSI Overbought Rejection")
        
        # MACD Crossover
        if latest["MACD_12_26_9"] > latest["MACDs_12_26_9"] and \
           df.iloc[-2]["MACD_12_26_9"] < df.iloc[-2]["MACDs_12_26_9"]:
            signals.append("MACD Bullish Cross")
        
        # ML Prediction
        ml_signal = generate_ml_signal(df)
        if ml_signal:
            signals.append(f"ML Prediction: {ml_signal}")
        
        return signals
    
    except Exception as e:
        app.logger.error(f"Signal error: {str(e)}")
        return []

@app.route("/")
def main_endpoint():
    try:
        df = get_futures_kline()
        if df is None or len(df) < 50:
            return jsonify({"status": "error", "message": "Insufficient data"}), 500

        df = calculate_indicators(df)
        if df is None:
            return jsonify({"status": "error", "message": "Indicator failure"}), 500

        return jsonify({
            "status": "success",
            "signals": generate_signals(df),
            "metrics": {
                "price": round(df.iloc[-1]["close"], 2),
                "rsi": round(df.iloc[-1]["rsi"], 2),
                "volatility": round(df.iloc[-1]["volatility"], 4),
                "atr": round(df.iloc[-1]["atr"], 2),
                "bb_upper": round(df.iloc[-1]["BBU_20_2.0"], 2),
                "bb_lower": round(df.iloc[-1]["BBL_20_2.0"], 2)
            },
            "ml_ready": model is not None
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
