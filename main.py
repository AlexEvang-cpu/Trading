import os
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import traceback
from flask import Flask, jsonify
from datetime import datetime
from joblib import load
from apscheduler.schedulers.background import BackgroundScheduler  # NEW

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))
MODEL_PATH = os.getenv("MODEL_PATH", "price_model.joblib")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # NEW
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")      # NEW

# Enhanced Trading parameters
TRADE_PARAMS = {
    "risk_reward_ratio": 2.0,
    "max_position_size": 0.08,
    "stop_loss_pct": 1.2,
    "take_profit_pct": 2.5
}

# Signal weights for confidence calculation
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

# ML Initialization
try:
    from sklearn.ensemble import RandomForestClassifier
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    app.logger.warning("ML features disabled - scikit-learn not installed")

model = load(MODEL_PATH) if ML_ENABLED and os.path.exists(MODEL_PATH) else None

# Initialize scheduler
scheduler = BackgroundScheduler()  # NEW

def send_telegram_alert(message):  # NEW FUNCTION
    """Send trading alerts to Telegram with error handling"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        app.logger.warning("Telegram credentials missing - notifications disabled")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        app.logger.error(f"Telegram send failed: {str(e)}")
        return False

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

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit='s', utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors='coerce'),
            "high": pd.to_numeric(raw_data["realHigh"], errors='coerce'),
            "low": pd.to_numeric(raw_data["realLow"], errors='coerce'),
            "close": pd.to_numeric(raw_data["realClose"], errors='coerce'),
            "volume": pd.to_numeric(raw_data["vol"], errors='coerce')
        })
        
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
    """Enhanced technical analysis with dynamic parameters"""
    step = "INDICATOR_CALCULATION"
    try:
        log_step(step, "Starting indicator calculation", {"input_shape": df.shape})
        
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
        
        df["hour"] = df["timestamp"].dt.hour
        df["session"] = np.select(
            [
                df["hour"].between(8, 11),
                df["hour"].between(12, 15)
            ],
            ["morning", "afternoon"],
            default="other"
        )
        
        required_columns = [
            'close', 'rsi', 'rsi_ma', 'ema_20', 'ema_50', 
            'bb_BBU_20_2.0', 'bb_BBL_20_2.0', 'atr',
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
    """Enhanced signal generation with dynamic thresholds"""
    step = "SIGNAL_GENERATION"
    try:
        log_step(step, "Starting signal generation")
        latest = df.iloc[-1]
        signals = []
        
        price_change_5m = latest["close"] - df.iloc[-5]["close"]
        change_pct = price_change_5m / df.iloc[-5]["close"]
        signals.append(f"5m Change: {price_change_5m:+.2f} ({change_pct:.2%})")
        
        bb_upper = latest["bb_BBU_20_2.0"]
        bb_lower = latest["bb_BBL_20_2.0"]
        
        if latest["close"] > bb_upper * 0.995:
            signals.append(f"BB Approach Upper ({latest['close']:.2f} > {bb_upper*0.995:.2f})")
        elif latest["close"] < bb_lower * 1.005:
            signals.append(f"BB Approach Lower ({latest['close']:.2f} < {bb_lower*1.005:.2f})")
        
        trend_direction = 1 if latest["close"] > latest["ema_20"] else -1
        rsi_buy_threshold = 40 + (5 * trend_direction)
        rsi_sell_threshold = 60 + (5 * trend_direction)
        
        if latest["rsi"] < rsi_buy_threshold and latest["rsi"] > latest["rsi_ma"]:
            signals.append(f"RSI Bullish ({latest['rsi']:.1f} < {rsi_buy_threshold})")
        elif latest["rsi"] > rsi_sell_threshold and latest["rsi"] < latest["rsi_ma"]:
            signals.append(f"RSI Bearish ({latest['rsi']:.1f} > {rsi_sell_threshold})")
            
        volume_ratio = latest["volume"] / latest["volume_ma"]
        if latest["volume_spike"] == 1:
            signals.append(f"Strong Volume Spike ({volume_ratio:.1f}x MA)")
            
        if latest["session"] == "morning" and latest["hour"] < 12:
            signals.append("Morning Trend Potential")
        elif latest["session"] == "afternoon" and latest["hour"] < 16:
            signals.append("Afternoon Trend Potential")
            
        if model is not None:
            try:
                features = df[[
                    'rsi', 'close', 'bb_BBU_20_2.0', 'bb_BBL_20_2.0',
                    'volume_ma', 'hour', 'atr'
                ]].iloc[-1].values.reshape(1, -1)
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                confidence = max(proba)
                signals.append(f"ML: {'Bullish' if prediction == 1 else 'Bearish'} ({confidence:.1%})")
            except Exception as ml_error:
                log_step(step, f"ML prediction failed: {str(ml_error)}", 'error')

        log_step(step, f"Generated {len(signals)} signals", signals)
        return signals

    except Exception as e:
        log_step(step, f"Signal generation failed: {str(e)}", traceback.format_exc(), 'error')
        return []

def generate_trading_decision(signals, metrics):
    """Enhanced decision making with weighted confidence"""
    decision = {
        "action": "hold",
        "confidence": 0.0,
        "signal_strength": {
            "bullish": 0.0,
            "bearish": 0.0,
            "neutral": 0.0
        },
        "detailed_signals": [],
        "risk_parameters": {
            "stop_loss": None,
            "take_profit": None,
            "position_size": TRADE_PARAMS["max_position_size"],
            "risk_reward_ratio": TRADE_PARAMS["risk_reward_ratio"]
        }
    }
    
    current_price = metrics["price"]
    atr = metrics["atr"]
    
    for signal in signals:
        decision["detailed_signals"].append(signal)
        weight = 1.0
        for key in SIGNAL_WEIGHTS:
            if key in signal:
                weight = SIGNAL_WEIGHTS[key]
                break
                
        if any(kw in signal for kw in ["Bullish", "Upper", "Buy"]):
            decision["signal_strength"]["bullish"] += weight
        elif any(kw in signal for kw in ["Bearish", "Lower", "Sell"]):
            decision["signal_strength"]["bearish"] += weight
        else:
            decision["signal_strength"]["neutral"] += weight

    total_strength = sum(decision["signal_strength"].values())
    if total_strength > 0:
        rsi_factor = 1 - abs(metrics["rsi"] - 50)/50
        raw_confidence = max(
            decision["signal_strength"]["bullish"],
            decision["signal_strength"]["bearish"]
        ) / total_strength
        
        decision["confidence"] = round((raw_confidence * 0.7 + rsi_factor * 0.3) * 100, 1)
        
        if decision["signal_strength"]["bullish"] > decision["signal_strength"]["bearish"]:
            if decision["confidence"] >= 55:
                decision["action"] = "long"
                decision["risk_parameters"]["stop_loss"] = round(current_price - atr * 1.2, 2)
                decision["risk_parameters"]["take_profit"] = round(current_price + atr * TRADE_PARAMS["risk_reward_ratio"], 2)
        elif decision["signal_strength"]["bearish"] > decision["signal_strength"]["bullish"]:
            if decision["confidence"] >= 55:
                decision["action"] = "short"
                decision["risk_parameters"]["stop_loss"] = round(current_price + atr * 1.2, 2)
                decision["risk_parameters"]["take_profit"] = round(current_price - atr * TRADE_PARAMS["risk_reward_ratio"], 2)

    if decision["action"] != "hold":
        volatility_ratio = atr / current_price
        confidence_factor = decision["confidence"] / 100
        decision["risk_parameters"]["position_size"] = round(
            TRADE_PARAMS["max_position_size"] * confidence_factor * (1 - min(volatility_ratio, 0.3)),
            4
        )

    return decision

@app.route("/")
def trading_dashboard():
    """Main trading endpoint with full metrics"""
    try:
        log_step("SYSTEM", "Processing pipeline started")
        
        raw_data = get_futures_kline()
        if raw_data is None:
            return jsonify({
                "status": "error",
                "stage": "data_fetch",
                "message": "Failed to retrieve market data",
                "advice": ["Check API status", "Verify symbol/interval"]
            }), 503

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

        signals = generate_signals(processed_data)
        latest = processed_data.iloc[-1]
        metrics = {
            "price": latest["close"],
            "rsi": latest["rsi"],
            "rsi_ma": latest["rsi_ma"],
            "ema_20": latest["ema_20"],
            "ema_50": latest["ema_50"],
            "bb_upper": latest["bb_BBU_20_2.0"],
            "bb_lower": latest["bb_BBL_20_2.0"],
            "atr": latest["atr"],
            "volume": latest["volume"],
            "volume_ma": latest["volume_ma"],
            "volume_spike": bool(latest["volume_spike"]),
            "session": latest["session"],
            "hour": int(latest["hour"])
        }

        response_data = {
            "status": "success",
            "decision": generate_trading_decision(signals, metrics),
            "metrics": metrics,
            "context": {
                "interval": DEFAULT_INTERVAL,
                "symbol": DEFAULT_SYMBOL,
                "model_active": model is not None
            }
        }

        # Telegram notification logic  # NEW
        if response_data["decision"]["action"] != "hold":
            message = (
                f"🚨 *{DEFAULT_SYMBOL} Trade Signal* 🚨\n"
                f"Action: {response_data['decision']['action'].upper()}\n"
                f"Confidence: {response_data['decision']['confidence']}%\n"
                f"Price: {metrics['price']}\n"
                f"SL: {response_data['decision']['risk_parameters']['stop_loss']}\n"
                f"TP: {response_data['decision']['risk_parameters']['take_profit']}\n"
                f"RSI: {metrics['rsi']:.1f}\n"
                f"Volume: {metrics['volume']}"
            )
            send_telegram_alert(message)

        return jsonify(response_data)

    except Exception as e:
        log_step("SYSTEM", f"Unhandled exception: {str(e)}", traceback.format_exc(), 'error')
        return jsonify({
            "status": "error",
            "message": "System failure",
            "error_details": str(e)[:200]
        }), 500

def scheduled_update():
    """Direct data fetch without HTTP client"""
    try:
        raw_data = get_futures_kline()
        processed_data = calculate_indicators(raw_data)
        latest = processed_data.iloc[-1]
        
        message = (
            f"⏰ *{DEFAULT_SYMBOL} Status Update*\n"
            f"Price: {latest['close']}\n"
            f"RSI: {latest['rsi']:.1f}\n"
            f"Volume: {latest['volume']}\n"
            f"Session: {latest['session'].title()}"
        )
        send_telegram_alert(message)
        
    except Exception as e:
        app.logger.error(f"Scheduled update error: {str(e)}")

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

def send_full_indicators():
    """Send complete technical analysis snapshot to Telegram"""
    try:
        raw_data = get_futures_kline()
        if raw_data is None:
            return

        processed_data = calculate_indicators(raw_data)
        if processed_data is None:
            return

        latest = processed_data.iloc[-1]
        
        # Format indicator message
        message = (
            f"📊 *{DEFAULT_SYMBOL} Technical Snapshot* 📊\n"
            f"⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"• Price: ${latest['close']:.2f}\n"
            f"• 5m Change: {(latest['close'] - processed_data.iloc[-5]['close']):+.2f}\n"
            f"• RSI: {latest['rsi']:.1f} / MA: {latest['rsi_ma']:.1f}\n"
            f"• EMA20/50: {latest['ema_20']:.2f} | {latest['ema_50']:.2f}\n"
            f"• BB Width: {latest['bb_width']:.2f} (U: {latest['bb_BBU_20_2.0']:.2f} L: {latest['bb_BBL_20_2.0']:.2f})\n"
            f"• ATR: {latest['atr']:.2f}\n"
            f"• Volume: {latest['volume']:,.0f} (MA: {latest['volume_ma']:,.0f})\n"
            f"• Session: {latest['session'].title()} (Hour {latest['hour']})"
        )
        
        send_telegram_alert(message)

    except Exception as e:
        app.logger.error(f"Indicator report failed: {str(e)}")

# Add at the BOTTOM of your code (last 10 lines)
def initialize_scheduler():
    """Safe scheduler initialization for production"""
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true" and not app.debug:
        scheduler.add_job(scheduled_update, 'interval', minutes=5)
        scheduler.add_job(send_full_indicators, 'interval', minutes=5)
        scheduler.start()
        app.logger.info("Scheduler started with 2 jobs")

if __name__ == "__main__":
    initialize_scheduler()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
else:
    initialize_scheduler()  # For production environments like Railway
