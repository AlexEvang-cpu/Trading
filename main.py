import os
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import traceback
import json
from flask import Flask, jsonify, request
from datetime import datetime
from joblib import load
from apscheduler.schedulers.background import BackgroundScheduler

# ====================== CONFIGURATION ====================== #
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))
MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ===================== TRADING PARAMETERS ===================== #
TRADE_PARAMS = {
    "risk_reward_ratio": 2.0,
    "max_position_size": 0.08,
    "stop_loss_pct": 1.2,
    "take_profit_pct": 2.5
}

SIGNAL_WEIGHTS = {
    "BB Breakout": 1.7,
    "BB Breakdown": 1.7,
    "RSI Oversold": 1.4,
    "RSI Overbought": 1.4,
    "Volume Spike": 1.3,
    "Session Trend": 1.2,
    "ML Prediction": 2.0
}

app = Flask(__name__)
app.logger.setLevel('DEBUG')

# ====================== ML INITIALIZATION ====================== #
try:
    from sklearn.ensemble import RandomForestClassifier
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    app.logger.warning("ML features disabled - scikit-learn not installed")

model = load(MODEL_PATH) if ML_ENABLED and os.path.exists(MODEL_PATH) else None

# ====================== SCHEDULER SETUP ====================== #
scheduler = BackgroundScheduler()

# ====================== CORE TRADING FUNCTIONS ====================== #
def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch market data from MEXC Futures API"""
    try:
        endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
        response = requests.get(endpoint, params={"interval": interval, "limit": limit}, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            return None

        raw_data = data.get("data", {})
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit='s', utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors='coerce'),
            "high": pd.to_numeric(raw_data["realHigh"], errors='coerce'),
            "low": pd.to_numeric(raw_data["realLow"], errors='coerce'),
            "close": pd.to_numeric(raw_data["realClose"], errors='coerce'),
            "volume": pd.to_numeric(raw_data["vol"], errors='coerce')
        })
        return df.dropna().iloc[-250:]
    except Exception as e:
        app.logger.error(f"Data fetch error: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    try:
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["rsi_ma"] = df["rsi"].rolling(14).mean()
        df["ema_20"] = ta.ema(df["close"], length=20)
        df["ema_50"] = ta.ema(df["close"], length=50)
        
        bbands = ta.bbands(df["close"], length=20, std=2)
        df = pd.concat([df, bbands.add_prefix("bb_")], axis=1)
        df["bb_width"] = df["bb_BBU_20_2.0"] - df["bb_BBL_20_2.0"]
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_spike"] = (df["volume"] > 2.0 * df["volume_ma"]).astype(int)
        return df.iloc[-100:]
    except Exception as e:
        app.logger.error(f"Indicator error: {str(e)}")
        return None

def generate_signals(df):
    """Generate trading signals"""
    try:
        latest = df.iloc[-1]
        signals = []
        
        # Bollinger Bands
        if latest["close"] > latest["bb_BBU_20_2.0"]:
            signals.append("BB Breakout")
        elif latest["close"] < latest["bb_BBL_20_2.0"]:
            signals.append("BB Breakdown")
            
        # RSI Logic
        if latest["rsi"] < 35:
            signals.append("RSI Oversold")
        elif latest["rsi"] > 65:
            signals.append("RSI Overbought")
            
        # Volume Spike
        if latest["volume_spike"] == 1:
            signals.append("Volume Spike")
            
        # ML Prediction
        if model:
            features = df[[
                'rsi', 'close', 'bb_BBU_20_2.0', 'bb_BBL_20_2.0',
                'volume_ma', 'atr'
            ]].iloc[-1].values.reshape(1, -1)
            prediction = model.predict(features)[0]
            signals.append(f"ML Prediction: {'Bullish' if prediction == 1 else 'Bearish'}")
            
        return signals
    except Exception as e:
        app.logger.error(f"Signal error: {str(e)}")
        return []

def generate_trading_decision(signals, metrics):
    """Generate trading decision with risk management"""
    decision = {
        "action": "hold",
        "confidence": 0.0,
        "risk_parameters": {
            "stop_loss": None,
            "take_profit": None,
            "position_size": TRADE_PARAMS["max_position_size"]
        }
    }
    
    if not signals:
        return decision
        
    # Simplified decision logic
    bullish = sum(1 for s in signals if "Breakout" in s or "Oversold" in s or "Bullish" in s)
    bearish = sum(1 for s in signals if "Breakdown" in s or "Overbought" in s or "Bearish" in s)
    
    confidence = min(100 * (abs(bullish - bearish) / len(signals)), 100)
    
    if bullish > bearish and confidence >= 55:
        decision.update({
            "action": "long",
            "confidence": round(confidence, 1),
            "risk_parameters": {
                "stop_loss": metrics["price"] * (1 - TRADE_PARAMS["stop_loss_pct"]/100),
                "take_profit": metrics["price"] * (1 + TRADE_PARAMS["take_profit_pct"]/100),
                "position_size": TRADE_PARAMS["max_position_size"] * confidence/100
            }
        })
    elif bearish > bullish and confidence >= 55:
        decision.update({
            "action": "short",
            "confidence": round(confidence, 1),
            "risk_parameters": {
                "stop_loss": metrics["price"] * (1 + TRADE_PARAMS["stop_loss_pct"]/100),
                "take_profit": metrics["price"] * (1 - TRADE_PARAMS["take_profit_pct"]/100),
                "position_size": TRADE_PARAMS["max_position_size"] * confidence/100
            }
        })
    
    return decision

# ====================== API ENDPOINTS ====================== #
@app.route("/")
def trading_dashboard():
    """Main trading endpoint"""
    try:
        raw_data = get_futures_kline()
        if raw_data is None:
            return jsonify({"status": "error", "message": "Data fetch failed"}), 500
            
        processed_data = calculate_indicators(raw_data)
        if processed_data is None:
            return jsonify({"status": "error", "message": "Processing failed"}), 500
            
        latest = processed_data.iloc[-1]
        signals = generate_signals(processed_data)
        
        return jsonify({
            "status": "success",
            "symbol": DEFAULT_SYMBOL,
            "price": latest["close"],
            "signals": signals,
            "decision": generate_trading_decision(signals, {
                "price": latest["close"],
                "rsi": latest["rsi"],
                "atr": latest["atr"]
            }),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "development")
    })

# ====================== SCHEDULED TASKS ====================== #
def forced_update():
    """Send regular updates to Telegram"""
    try:
        raw_data = get_futures_kline()
        processed_data = calculate_indicators(raw_data)
        latest = processed_data.iloc[-1]
        
        message = (
            f"📊 {DEFAULT_SYMBOL} Update\n"
            f"Price: {latest['close']:.2f}\n"
            f"RSI: {latest['rsi']:.1f}\n"
            f"24h Change: {(latest['close'] - processed_data.iloc[0]['close']):.2f}"
        )
        send_telegram_alert(message)
    except Exception as e:
        app.logger.error(f"Scheduled update failed: {str(e)}")

def send_telegram_alert(message):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown"
            },
            timeout=5
        )
    except Exception as e:
        app.logger.error(f"Telegram send failed: {str(e)}")

# ====================== INITIALIZATION ====================== #
if __name__ == "__main__":
    scheduler.add_job(forced_update, 'interval', minutes=5)
    scheduler.start()
    
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
