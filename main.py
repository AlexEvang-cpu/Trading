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

# ... [Keep the existing get_futures_kline and calculate_indicators functions] ...

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
    atr = metrics["bb_width"] / 2  # Use half Bollinger Band width as volatility measure
    
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
