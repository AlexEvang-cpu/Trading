import os
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import traceback
from flask import Flask, jsonify
from datetime import datetime
from joblib import load

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")  # 1-minute timeframe
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))       # 5 hours of data
MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")

app = Flask(__name__)
app.logger.setLevel('DEBUG')

# ML Initialization
try:
    from sklearn.ensemble import RandomForestClassifier
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    app.logger.warning("ML features disabled - scikit-learn not installed")

model = load(MODEL_PATH) if ML_ENABLED and os.path.exists(MODEL_PATH) else None

def log_step(step, message, data=None, level='debug'):
    """Structured logging with data sanitization"""
    logger = getattr(app.logger, level)
    log_msg = f"[{step}] {message}"
    if data is not None:
        sanitized = str(data).replace('\n', ' ')[:200]
        log_msg += f" | Data: {sanitized}"
    logger(log_msg)

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch 1-minute market data with enhanced debugging"""
    step = "DATA_FETCH"
    try:
        log_step(step, "Initiating API request", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })
        
        endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
        response = requests.get(endpoint, params={"interval": interval, "limit": limit}, timeout=15)
        response.raise_for_status()
        
        log_step(step, f"API response received - Status: {response.status_code}")
        data = response.json()
        
        if not data.get("success", False):
            log_step(step, "API response indicates failure", data, 'error')
            return None

        raw_data = data.get("data", {})
        if not raw_data.get("time"):
            log_step(step, "Missing time data in response", raw_data, 'error')
            return None

        # Construct DataFrame with type safety
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit='s', utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors='coerce'),
            "high": pd.to_numeric(raw_data["realHigh"], errors='coerce'),
            "low": pd.to_numeric(raw_data["realLow"], errors='coerce'),
            "close": pd.to_numeric(raw_data["realClose"], errors='coerce'),
            "volume": pd.to_numeric(raw_data["vol"], errors='coerce')
        })
        
        # Data validation pipeline
        initial_count = len(df)
        df = df.dropna().reset_index(drop=True)
        log_step(step, f"Data cleaning complete - Remaining: {len(df)}/{initial_count} rows")
        
        if len(df) < 100:
            log_step(step, "Insufficient data after cleaning", {"min_required": 100, "actual": len(df)}, 'error')
            return None
            
        log_step(step, "Data sample", df.iloc[:3].to_dict('records'))
        return df.iloc[-250:]  # Last 4+ hours of data

    except Exception as e:
        log_step(step, f"Critical error: {str(e)}", traceback.format_exc(), 'error')
        return None

def calculate_indicators(df):
    """Advanced technical analysis with validation gates"""
    step = "INDICATOR_CALCULATION"
    try:
        log_step(step, "Starting indicator calculation", {"input_shape": df.shape})
        
        # Core Indicators
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        
        # Bollinger Bands with adaptive volatility
        bbands = ta.bbands(df["close"], length=14, std=1.5)
        df = pd.concat([df, bbands.add_prefix("bb_")], axis=1)
        df["bb_width"] = df["bb_BBU_14_1.5"] - df["bb_BBL_14_1.5"]
        
        # Volume analysis
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_spike"] = (df["volume"] > 1.5 * df["volume_ma"]).astype(int)
        
        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["session"] = np.select(
            [
                df["hour"].between(9, 12),
                df["hour"].between(13, 16)
            ],
            ["morning", "afternoon"],
            default="other"
        )
        
        # Validate calculations
        required_columns = [
            'close', 'rsi', 'ema_20', 'ema_50', 
            'bb_BBU_14_1.5', 'bb_BBL_14_1.5',
            'volume_ma', 'volume_spike', 'hour'
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            log_step(step, f"Missing critical columns: {missing}", 'error')
            return None
            
        log_step(step, "Indicator calculation successful", {
            "final_shape": df.shape,
            "columns": df.columns.tolist()
        })
        return df.iloc[-100:]  # Use most recent 100 periods

    except Exception as e:
        log_step(step, f"Calculation failed: {str(e)}", traceback.format_exc(), 'error')
        return None

def generate_signals(df):
    """Multi-factor signal generation with ML integration"""
    step = "SIGNAL_GENERATION"
    try:
        log_step(step, "Starting signal generation")
        latest = df.iloc[-1]
        signals = []
        
        # Price Action Signals
        price_change_5m = latest["close"] - df.iloc[-5]["close"]
        signals.append(f"5m Change: {'+' if price_change_5m >=0 else ''}{price_change_5m:.2f}")
        
        # Bollinger Band Signals
        bb_upper = latest["bb_BBU_14_1.5"]
        bb_lower = latest["bb_BBL_14_1.5"]
        if latest["close"] > bb_upper:
            signals.append("BB Breakout")
        elif latest["close"] < bb_lower:
            signals.append("BB Breakdown")
        
        # RSI + EMA Convergence
        if latest["rsi"] < 35 and latest["close"] > latest["ema_20"]:
            signals.append("RSI Oversold Bounce")
        elif latest["rsi"] > 65 and latest["close"] < latest["ema_20"]:
            signals.append("RSI Overbought Rejection")
            
        # Volume Spike Alert
        if latest["volume_spike"] == 1:
            signals.append("Volume Spike Detected")
            
        # Session-Based Pattern
        if latest["session"] == "morning":
            signals.append("Morning Session Trend")
            
        # ML Prediction
        if model is not None:
            try:
                features = df[[
                    'rsi', 'close', 'bb_BBU_14_1.5', 'bb_BBL_14_1.5',
                    'volume_ma', 'hour'
                ]].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(features)[0]
                signals.append(f"ML: {'Bullish' if prediction == 1 else 'Bearish'}")
            except Exception as ml_error:
                log_step(step, f"ML prediction failed: {str(ml_error)}", 'error')

        log_step(step, f"Generated {len(signals)} signals", signals)
        return signals

    except Exception as e:
        log_step(step, f"Signal generation failed: {str(e)}", traceback.format_exc(), 'error')
        return []

@app.route("/")
def trading_dashboard():
    """Main trading endpoint with full observability"""
    try:
        log_step("SYSTEM", "Processing pipeline started")
        
        # Data Acquisition
        raw_data = get_futures_kline()
        if raw_data is None:
            return jsonify({
                "status": "error",
                "stage": "data_fetch",
                "message": "Failed to retrieve market data",
                "advice": ["Check API status", "Verify symbol/interval"]
            }), 503

        # Technical Analysis
        processed_data = calculate_indicators(raw_data)
        if processed_data is None:
            return jsonify({
                "status": "error",
                "stage": "indicators",
                "data_stats": {
                    "initial_rows": len(raw_data),
                    "columns": raw_data.columns.tolist(),
                    "last_timestamp": raw_data["timestamp"].iloc[-1].isoformat()
                }
            }), 500

        # Signal Generation
        signals = generate_signals(processed_data)
        latest = processed_data.iloc[-1]

        return jsonify({
            "status": "success",
            "signals": signals,
            "metrics": {
                "price": latest["close"],
                "rsi": latest["rsi"],
                "ema_20": latest["ema_20"],
                "bb_upper": latest["bb_BBU_14_1.5"],
                "bb_lower": latest["bb_BBL_14_1.5"],
                "volume": latest["volume"]
            },
            "context": {
                "interval": DEFAULT_INTERVAL,
                "symbol": DEFAULT_SYMBOL,
                "model_active": model is not None
            }
        })

    except Exception as e:
        log_step("SYSTEM", f"Unhandled exception: {str(e)}", traceback.format_exc(), 'error')
        return jsonify({
            "status": "error",
            "message": "System failure",
            "error_details": str(e)[:200]
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
