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
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))
MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")

# Trading parameters
TRADE_PARAMS = {
    "risk_reward_ratio": 2.0,
    "max_position_size": 0.05,
    "stop_loss_pct": 1.0,
    "take_profit_pct": 2.0
}

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
        return df.iloc[-250:]

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
        return df.iloc[-100:]

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
        signals.append(f"5m Change: {price_change_5m:+.2f} ({price_change_5m/df.iloc[-5]['close']:.2%})")
        
        # Bollinger Band Signals
        bb_upper = latest["bb_BBU_14_1.5"]
        bb_lower = latest["bb_BBL_14_1.5"]
        if latest["close"] > bb_upper:
            signals.append(f"BB Breakout (Price: {latest['close']:.2f} > Upper: {bb_upper:.2f})")
        elif latest["close"] < bb_lower:
            signals.append(f"BB Breakdown (Price: {latest['close']:.2f} < Lower: {bb_lower:.2f})")
        
        # RSI + EMA Convergence
        rsi_threshold = 35 if latest["close"] > latest["ema_20"] else 65
        if latest["rsi"] < 35 and latest["close"] > latest["ema_20"]:
            signals.append(f"RSI Oversold Bounce (RSI: {latest['rsi']:.1f}, EMA20: {latest['ema_20']:.2f})")
        elif latest["rsi"] > 65 and latest["close"] < latest["ema_20"]:
            signals.append(f"RSI Overbought Rejection (RSI: {latest['rsi']:.1f}, EMA20: {latest['ema_20']:.2f})")
            
        # Volume Spike Alert
        volume_ratio = latest["volume"] / latest["volume_ma"]
        if latest["volume_spike"] == 1:
            signals.append(f"Volume Spike ({volume_ratio:.1f}x MA)")
            
        # Session-Based Pattern
        if latest["session"] == "morning":
            signals.append("Morning Session Trend Detected")
            
        # ML Prediction
        if model is not None:
            try:
                features = df[[
                    'rsi', 'close', 'bb_BBU_14_1.5', 'bb_BBL_14_1.5',
                    'volume_ma', 'hour'
                ]].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                confidence = max(proba)
                signals.append(f"ML: {'Bullish' if prediction == 1 else 'Bearish'} ({confidence:.1%} confidence)")
            except Exception as ml_error:
                log_step(step, f"ML prediction failed: {str(ml_error)}", 'error')

        log_step(step, f"Generated {len(signals)} signals", signals)
        return signals

    except Exception as e:
        log_step(step, f"Signal generation failed: {str(e)}", traceback.format_exc(), 'error')
        return []

def generate_trading_decision(signals, metrics):
    """Convert signals into executable trading decisions"""
    decision = {
        "action": "hold",
        "confidence": 0.0,
        "signal_strength": {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0
        },
        "detailed_signals": [],
        "risk_parameters": {
            "stop_loss": None,
            "take_profit": None,
            "position_size": TRADE_PARAMS["max_position_size"],
            "risk_reward_ratio": TRADE_PARAMS["risk_reward_ratio"]
        }
    }
    
    # Calculate volatility-adjusted parameters
    current_price = metrics["price"]
    atr = metrics["bb_width"] / 2
    
    for signal in signals:
        decision["detailed_signals"].append(signal)
        
        # Signal strength scoring
        if any(keyword in signal for keyword in ["Breakout", "Bounce", "Bullish"]):
            decision["signal_strength"]["bullish"] += 1
        elif any(keyword in signal for keyword in ["Breakdown", "Rejection", "Bearish"]):
            decision["signal_strength"]["bearish"] += 1
        else:
            decision["signal_strength"]["neutral"] += 1

    # Calculate total strength
    total = sum(decision["signal_strength"].values())
    if total > 0:
        decision["confidence"] = round(
            (max(decision["signal_strength"]["bullish"], decision["signal_strength"]["bearish"]) / total) * 100, 
            1
        )
        
        # Determine action
        if decision["signal_strength"]["bullish"] > decision["signal_strength"]["bearish"]:
            decision["action"] = "long"
            decision["risk_parameters"]["stop_loss"] = round(current_price - atr, 2)
            decision["risk_parameters"]["take_profit"] = round(current_price + (atr * TRADE_PARAMS["risk_reward_ratio"]), 2)
        elif decision["signal_strength"]["bearish"] > decision["signal_strength"]["bullish"]:
            decision["action"] = "short"
            decision["risk_parameters"]["stop_loss"] = round(current_price + atr, 2)
            decision["risk_parameters"]["take_profit"] = round(current_price - (atr * TRADE_PARAMS["risk_reward_ratio"]), 2)

    # Adjust position size based on volatility
    if decision["action"] != "hold":
        volatility_factor = atr / current_price
        decision["risk_parameters"]["position_size"] = round(
            TRADE_PARAMS["max_position_size"] * (1 - min(volatility_factor, 0.5)), 
            4
        )

    return decision

@app.route("/")
def trading_dashboard():
    """Main trading endpoint with full metrics"""
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

        # Generate outputs
        signals = generate_signals(processed_data)
        latest = processed_data.iloc[-1]
        metrics = {
            "price": latest["close"],
            "rsi": latest["rsi"],
            "ema_20": latest["ema_20"],
            "ema_50": latest["ema_50"],
            "bb_upper": latest["bb_BBU_14_1.5"],
            "bb_lower": latest["bb_BBL_14_1.5"],
            "bb_width": latest["bb_width"],
            "volume": latest["volume"],
            "volume_ma": latest["volume_ma"],
            "volume_spike": bool(latest["volume_spike"]),
            "session": latest["session"],
            "hour": int(latest["hour"])
        }

        return jsonify({
            "status": "success",
            "decision": generate_trading_decision(signals, metrics),
            "metrics": metrics,
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

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "python_version": os.sys.version,
            "platform": os.sys.platform
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
