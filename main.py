import os
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, jsonify
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")  # 1-minute timeframe
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))  # 5 hours of data
MODEL_PATH = "price_model.joblib"

app = Flask(__name__)

# Try loading ML model
try:
    model = load(MODEL_PATH)
except:
    model = None

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch and process 1-minute market data with enhanced validation"""
    endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    
    try:
        response = requests.get(endpoint, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            app.logger.error(f"MEXC API Error: {data.get('code', 'Unknown')}")
            return None

        raw_data = data.get("data", {})
        if not raw_data.get("time", []):
            app.logger.error("Empty data response from API")
            return None

        # Create DataFrame from MEXC's structured format
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit="s", utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors="coerce"),
            "high": pd.to_numeric(raw_data["realHigh"], errors="coerce"),
            "low": pd.to_numeric(raw_data["realLow"], errors="coerce"),
            "close": pd.to_numeric(raw_data["realClose"], errors="coerce"),
            "volume": pd.to_numeric(raw_data["vol"], errors="coerce")
        }).dropna()

        if len(df) < 50:  # Require minimum 50 periods
            app.logger.error(f"Insufficient data: {len(df)}/50 periods")
            return None
            
        return df.iloc[-250:]  # Return last ~4 hours

    except Exception as e:
        app.logger.error(f"Data fetch failed: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate advanced technical indicators with triple validation"""
    try:
        # Core Indicators
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        
        # Bollinger Bands with volatility
        bbands = ta.bbands(df["close"], length=20, std=2)
        df = pd.concat([df, bbands.add_prefix("bb_")], axis=1)
        df["bb_width"] = df["bb_BBU_20_2.0"] - df["bb_BBL_20_2.0"]
        
        # Advanced Volatility Metrics
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["volatility"] = df["close"].rolling(20).std()
        
        # Momentum Indicators
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd.add_prefix("macd_")], axis=1)
        
        # Pattern Recognition
        df["engulfing"] = ta.cdl_engulfing(df["open"], df["high"], df["low"], df["close"])
        
        # Machine Learning Features
        df["volume_pct"] = df["volume"].pct_change(periods=3)
        df["price_change"] = df["close"].pct_change(periods=3)
        
        # Clean and validate
        df = df.dropna()
        required_cols = [
            "close", "rsi", "bb_BBU_20_2.0", "bb_BBL_20_2.0",
            "ema_20", "ema_50", "atr", "volatility", "macd_MACD_12_26_9"
        ]
        if not all(col in df.columns for col in required_cols):
            app.logger.error("Missing required columns in indicators")
            return None

        return df.iloc[-100:]  # Use most recent 100 periods

    except Exception as e:
        app.logger.error(f"Indicator calculation failed: {str(e)}")
        return None

def generate_ml_signal(df):
    """Generate machine learning prediction with fallback"""
    if model is None or len(df) < 50:
        return None
    
    try:
        # Prepare features matching training format
        features = df[[
            "rsi", "close", "volatility", 
            "macd_MACD_12_26_9", "volume_pct"
        ]].iloc[-1].values.reshape(1, -1)
        
        prediction = model.predict(features)
        proba = model.predict_proba(features)[0]
        return {
            "direction": "Bullish" if prediction[0] == 1 else "Bearish",
            "confidence": round(max(proba), 2)
        }
    except Exception as e:
        app.logger.error(f"ML prediction failed: {str(e)}")
        return None

def generate_signals(df):
    """Generate sophisticated trading signals with multiple confirmation"""
    try:
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-3]  # 3 periods back for momentum

        # 1. Volatility-based Signals
        volatility_ratio = latest["atr"] / latest["close"]
        if volatility_ratio > 0.005:
            signals.append("HighVol Entry")
        
        # 2. Bollinger Band Strategy
        if latest["close"] > latest["bb_BBU_20_2.0"]:
            signals.append("BB Breakout")
        elif latest["close"] < latest["bb_BBL_20_2.0"]:
            signals.append("BB Breakdown")
        
        # 3. RSI with EMA Filter
        if latest["rsi"] < 35 and latest["close"] > latest["ema_20"]:
            signals.append("RSI Oversold (EMA Support)")
        elif latest["rsi"] > 65 and latest["close"] < latest["ema_20"]:
            signals.append("RSI Overbought (EMA Resistance)")

        # 4. MACD Crossover
        if (latest["macd_MACD_12_26_9"] > latest["macd_MACDs_12_26_9"] and
            df.iloc[-2]["macd_MACD_12_26_9"] < df.iloc[-2]["macd_MACDs_12_26_9"]):
            signals.append("MACD Bull Cross")

        # 5. ML Prediction
        ml_output = generate_ml_signal(df)
        if ml_output:
            signals.append(f"ML: {ml_output['direction']} ({ml_output['confidence']})")

        # 6. Trend Confirmation
        if len(signals) >= 3:
            signals.append("Strong Consensus")
            
        return signals

    except Exception as e:
        app.logger.error(f"Signal generation failed: {str(e)}")
        return []

@app.route("/")
def trading_signals():
    try:
        # Data Pipeline
        price_data = get_futures_kline()
        if price_data is None:
            return jsonify({"status": "error", "message": "Data unavailable"}), 503
        
        processed_data = calculate_indicators(price_data)
        if processed_data is None:
            return jsonify({"status": "error", "message": "Indicator failure"}), 500

        # Generate Output
        latest = processed_data.iloc[-1]
        return jsonify({
            "status": "success",
            "signals": generate_signals(processed_data),
            "metrics": {
                "price": round(latest["close"], 2),
                "rsi": round(latest["rsi"], 1),
                "bb_upper": round(latest["bb_BBU_20_2.0"], 2),
                "bb_lower": round(latest["bb_BBL_20_2.0"], 2),
                "volatility": round(latest["volatility"], 4),
                "atr": round(latest["atr"], 2),
                "ema_20": round(latest["ema_20"], 2)
            },
            "model_active": model is not None
        })

    except Exception as e:
        app.logger.critical(f"System failure: {str(e)}")
        return jsonify({"status": "error", "message": "Internal failure"}), 500

def train_model():
    """Train ML model on historical data (run separately)"""
    try:
        df = get_futures_kline(limit=5000)  # ~3 days of 1m data
        if df is None:
            raise ValueError("No data for training")
            
        df = calculate_indicators(df)
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        
        # Select features and target
        features = df[["rsi", "close", "volatility", "macd_MACD_12_26_9", "volume_pct"]].dropna()
        target = df.loc[features.index, "target"]
        
        # Train and save model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(features, target)
        dump(model, MODEL_PATH)
        print(f"Model trained with {len(features)} samples")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")

if __name__ == "__main__":
    # Train model first by uncommenting:
    # train_model()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
