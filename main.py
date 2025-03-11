import os
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import traceback
import json
from flask import Flask, jsonify, request  # Added 'request' import
from datetime import datetime
from joblib import load
from apscheduler.schedulers.background import BackgroundScheduler

# ====================== CONFIGURATION ====================== #
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))
MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")

# Telegram Configuration (CHANGE THESE IN YOUR .env FILE!)
TELEGRAM_BOT_TOKEN = os.getenv("7767920761:AAHLm9Lgs4UQpaUon04aPc1AVfKAgTtHep8")  # Get from @BotFather
TELEGRAM_CHAT_ID = os.getenv("5704086227")      # Use /getUpdates to find
TELEGRAM_WEBHOOK_SECRET = os.getenv("5f4dcc3b5aa765d61d8327dhb882cf99231f2a717d4c5d7c2f5x3c4f4f5b6a7x")# Generate random string

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

# ===================== TELEGRAM COMMANDS ===================== #
@app.route('/telegram', methods=['POST'])
def handle_telegram_commands():
    """Process real-time commands from Telegram"""
    try:
        # Security check
        if request.args.get('secret') != TELEGRAM_WEBHOOK_SECRET:
            return "Unauthorized", 401
            
        data = request.get_json()
        message = data.get('message', {})
        text = message.get('text', '').lower()
        chat_id = str(message.get('chat', {}).get('id'))
        
        # Validate authorized user
        if chat_id != TELEGRAM_CHAT_ID:
            return "OK", 200
            
        # Process commands
        response_text = "❌ Unknown command. Try:\n/price\n/metrics\n/signal\n/help"
        
        if text == '/price':
            latest = get_latest_metrics()
            response_text = f"📈 {DEFAULT_SYMBOL}\nCurrent Price: {latest['price']}"
            
        elif text == '/metrics':
            latest = get_latest_metrics()
            response_text = (f"📊 Metrics\nRSI: {latest['rsi']:.1f}\n"
                            f"Volume: {latest['volume']}\nATR: {latest['atr']:.2f}")
            
        elif text == '/signal':
            latest = get_latest_metrics()
            response_text = (f"🚨 Current Signal\nAction: {latest['action'].upper()}\n"
                            f"Confidence: {latest['confidence']}%\n"
                            f"SL: {latest['sl']}\nTP: {latest['tp']}")
            
        elif text == '/help':
            response_text = ("🛠 Available Commands:\n"
                            "/price - Current price\n"
                            "/metrics - Key metrics\n"
                            "/signal - Trading signal\n"
                            "/status - System health")
        
        send_telegram_alert(response_text, chat_id)
        
    except Exception as e:
        app.logger.error(f"Command error: {str(e)}")
        
    return "OK", 200

def get_latest_metrics():
    """Get fresh market data for commands"""
    raw_data = get_futures_kline()
    processed_data = calculate_indicators(raw_data)
    latest = processed_data.iloc[-1]
    signals = generate_signals(processed_data)
    decision = generate_trading_decision(signals, {
        "price": latest["close"],
        "rsi": latest["rsi"],
        "atr": latest["atr"]
    })
    
    return {
        "price": latest["close"],
        "rsi": latest["rsi"],
        "volume": latest["volume"],
        "atr": latest["atr"],
        "action": decision["action"],
        "confidence": decision["confidence"],
        "sl": decision["risk_parameters"]["stop_loss"],
        "tp": decision["risk_parameters"]["take_profit"]
    }

# ================== MANDATORY 5-MINUTE UPDATES ================== #
def forced_update():
    """Guaranteed notification every 5 minutes"""
    try:
        raw_data = get_futures_kline()
        processed_data = calculate_indicators(raw_data)
        latest = processed_data.iloc[-1] if processed_data is not None else None
        
        if latest is not None:
            message = (
                f"⏰ Mandatory Update ({datetime.utcnow().strftime('%H:%M UTC')})\n"
                f"• Price: {latest['close']}\n"
                f"• RSI: {latest['rsi']:.1f}\n"
                f"• Volume: {latest['volume']}\n"
                f"• ATR: {latest['atr']:.2f}"
            )
            send_telegram_alert(message)
            
    except Exception as e:
        app.logger.error(f"Forced update failed: {str(e)}")
        send_telegram_alert("🔴 Update failed - check logs")

# ================== ENHANCED NOTIFICATIONS ================== #
def send_telegram_alert(message, chat_id=None):
    """Improved notification system with retries"""
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        app.logger.error("Telegram credentials missing")
        return False

    for attempt in range(3):
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                },
                timeout=5
            )
            if response.status_code == 200:
                return True
        except Exception as e:
            app.logger.warning(f"Attempt {attempt+1}/3 failed: {str(e)}")
    
    app.logger.error("All Telegram send attempts failed")
    return False

# ================== MAIN TRADING LOGIC ================== #
# [Keep all existing functions below exactly as you provided them]
# get_futures_kline(), calculate_indicators(), generate_signals(), 
# generate_trading_decision(), trading_dashboard(), etc.

# ================== SCHEDULER INITIALIZATION ================== #
if __name__ == "__main__":
    # Set up scheduled jobs
    scheduler.add_job(forced_update, 'interval', minutes=5, id='forced_updates')
    scheduler.start()
    
    # Configure Telegram webhook
    if TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
                json={"url": f"{os.getenv('BASE_URL')}/telegram?secret={TELEGRAM_WEBHOOK_SECRET}"}
            )
        except Exception as e:
            app.logger.error(f"Webhook setup failed: {str(e)}")

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
