import os
import requests
import pandas as pd
import pandas_ta as ta
import traceback
from flask import Flask, jsonify
from datetime import datetime

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min1")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "300"))

app = Flask(__name__)
app.logger.setLevel('DEBUG')  # Enable verbose logging

def log_step(step, message, data=None):
    """Uniform logging format for debugging"""
    app.logger.debug(f"[{step}] {message}")
    if data is not None:
        app.logger.debug(f"[{step}] Data: {str(data)[:200]}...")  # Limit data length

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch market data with detailed debugging"""
    step = "DATA_FETCH"
    try:
        log_step(step, "Starting API call", {"symbol": symbol, "interval": interval, "limit": limit})
        
        endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
        params = {"interval": interval, "limit": limit}
        
        response = requests.get(endpoint, params=params, timeout=15)
        log_step(step, f"API response status: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        log_step(step, "Raw API response", data)

        if not data.get("success", False):
            log_step(step, "API success=False", data)
            return None

        raw_data = data.get("data", {})
        if not raw_data.get("time"):
            log_step(step, "Empty time data", raw_data)
            return None

        # DataFrame construction debugging
        log_step(step, "Building DataFrame")
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit="s", utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors="coerce"),
            "high": pd.to_numeric(raw_data["realHigh"], errors="coerce"),
            "low": pd.to_numeric(raw_data["realLow"], errors="coerce"),
            "close": pd.to_numeric(raw_data["realClose"], errors="coerce"),
            "volume": pd.to_numeric(raw_data["vol"], errors="coerce")
        })
        
        log_step(step, "DataFrame shape after creation", df.shape)
        
        # Data cleaning
        initial_count = len(df)
        df = df.dropna()
        log_step(step, f"Data cleaning: {initial_count} -> {len(df)} rows remaining")
        
        if len(df) < 50:
            log_step(step, f"Insufficient data after cleaning: {len(df)}/50")
            return None
            
        log_step(step, "Final data sample", df.iloc[:3].to_dict())
        return df.iloc[-250:]

    except Exception as e:
        log_step(step, f"Critical error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return None

def calculate_indicators(df):
    """Indicator calculation with step-by-step validation"""
    step = "INDICATORS"
    try:
        if df is None:
            log_step(step, "No input data")
            return None
            
        log_step(step, "Input data shape", df.shape)
        
        # RSI Calculation
        log_step(step, "Calculating RSI")
        df["rsi"] = ta.rsi(df["close"], length=14)
        if df["rsi"].isnull().all():
            log_step(step, "RSI calculation failed")
            return None

        # Bollinger Bands
        log_step(step, "Calculating Bollinger Bands")
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is None:
            log_step(step, "Bollinger Bands failed")
            return None
            
        df = pd.concat([df, bbands], axis=1)
        log_step(step, "Post-BBANDS columns", df.columns.tolist())

        # EMA Calculation
        log_step(step, "Calculating EMA")
        df["ema_20"] = ta.ema(df["close"], length=20)
        if df["ema_20"].isnull().all():
            log_step(step, "EMA calculation failed")
            return None

        # Validate final columns
        required_cols = ["close", "rsi", "BBU_20_2.0", "BBL_20_2.0", "ema_20"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            log_step(step, f"Missing columns: {missing}")
            return None

        log_step(step, "Final indicators sample", df[required_cols].iloc[-3:].to_dict())
        return df.iloc[-100:]

    except Exception as e:
        log_step(step, f"Calculation error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return None

def generate_signals(df):
    """Signal generation with detailed checks"""
    step = "SIGNALS"
    try:
        if df is None or df.empty:
            log_step(step, "No data for signals")
            return []

        latest = df.iloc[-1]
        log_step(step, "Latest data point", latest.to_dict())

        signals = []
        
        # Bollinger Bands Check
        bb_upper = latest["BBU_20_2.0"]
        bb_lower = latest["BBL_20_2.0"]
        price = latest["close"]
        
        log_step(step, f"BB Check: Price {price} vs [{bb_lower}, {bb_upper}]")
        if price > bb_upper:
            signals.append("Upper Band Breakout")
        elif price < bb_lower:
            signals.append("Lower Band Breakdown")

        # RSI Check
        rsi = latest["rsi"]
        ema = latest["ema_20"]
        log_step(step, f"RSI Check: {rsi} (EMA20: {ema})")
        if rsi < 35 and price > ema:
            signals.append("RSI Oversold Bounce")
        elif rsi > 65 and price < ema:
            signals.append("RSI Overbought Rejection")

        log_step(step, f"Generated signals: {signals}")
        return signals

    except Exception as e:
        log_step(step, f"Signal error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return []

@app.route("/debug")
def full_debug():
    """Special endpoint for complete debugging"""
    debug_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "symbol": DEFAULT_SYMBOL,
            "interval": DEFAULT_INTERVAL,
            "limit": DEFAULT_LIMIT
        }
    }
    
    try:
        # Step 1: Raw API Data
        raw_data = get_futures_kline()
        debug_info["raw_data"] = {
            "status": "success" if raw_data is not None else "failed",
            "shape": raw_data.shape if raw_data is not None else None,
            "sample": raw_data.iloc[:3].to_dict() if raw_data is not None else None
        }

        # Step 2: Indicators
        if raw_data is not None:
            indicators = calculate_indicators(raw_data)
            debug_info["indicators"] = {
                "status": "success" if indicators is not None else "failed",
                "columns": indicators.columns.tolist() if indicators is not None else None,
                "sample": indicators.iloc[:3].to_dict() if indicators is not None else None
            }
        else:
            debug_info["indicators"] = {"status": "skipped"}

        # Step 3: Signals
        if indicators is not None:
            signals = generate_signals(indicators)
            debug_info["signals"] = {
                "generated": signals,
                "latest_data": indicators.iloc[-1].to_dict() if not indicators.empty else None
            }
        else:
            debug_info["signals"] = {"status": "skipped"}

        return jsonify(debug_info)

    except Exception as e:
        debug_info["error"] = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        return jsonify(debug_info), 500

@app.route("/")
def main_handler():
    try:
        log_step("MAIN", "Process started")
        
        # Data Fetching
        df = get_futures_kline()
        if df is None:
            log_step("MAIN", "Data fetch failed")
            return jsonify({
                "status": "error",
                "stage": "data_fetch",
                "message": "Failed to retrieve market data",
                "advice": [
                    "Check API availability",
                    "Verify symbol/interval parameters",
                    "Ensure network connectivity"
                ]
            }), 503

        # Indicator Calculation
        processed_df = calculate_indicators(df)
        if processed_df is None:
            log_step("MAIN", "Indicator calculation failed")
            return jsonify({
                "status": "error",
                "stage": "indicators",
                "message": "Technical analysis failed",
                "data_stats": {
                    "initial_rows": len(df),
                    "columns": df.columns.tolist(),
                    "last_close": df["close"].iloc[-1] if not df.empty else None
                }
            }), 500

        # Signal Generation
        signals = generate_signals(processed_df)
        latest = processed_df.iloc[-1]

        return jsonify({
            "status": "success",
            "signals": signals,
            "metrics": {
                "price": latest["close"],
                "rsi": latest["rsi"],
                "bb_upper": latest["BBU_20_2.0"],
                "bb_lower": latest["BBL_20_2.0"],
                "ema_20": latest["ema_20"]
            },
            "debug": {
                "data_points": len(processed_df),
                "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else None
            }
        })

    except Exception as e:
        log_step("MAIN", f"Unhandled exception: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "stage": "unhandled",
            "message": "Critical system failure",
            "error_details": str(e),
            "traceback": traceback.format_exc()
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
