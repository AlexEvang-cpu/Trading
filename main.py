import os
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify
from datetime import datetime

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min15")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "100"))

app = Flask(__name__)

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Fetch and process futures market data from MEXC"""
    endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success", False):
            app.logger.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None

        raw_data = data.get("data", {})
        if not raw_data or "time" not in raw_data:
            app.logger.error("Invalid API response structure")
            return None

        # Create DataFrame from separated arrays
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(raw_data["time"], unit="s", utc=True),
            "open": pd.to_numeric(raw_data["realOpen"], errors="coerce"),
            "high": pd.to_numeric(raw_data["realHigh"], errors="coerce"),
            "low": pd.to_numeric(raw_data["realLow"], errors="coerce"),
            "close": pd.to_numeric(raw_data["realClose"], errors="coerce"),
            "volume": pd.to_numeric(raw_data["vol"], errors="coerce"),
            "amount": pd.to_numeric(raw_data["amount"], errors="coerce")
        })

        # Clean and validate data
        df = df.dropna()
        if df.empty or len(df) < 5:
            app.logger.error("Insufficient valid data after processing")
            return None
            
        return df

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error: {str(e)}")
    except Exception as e:
        app.logger.error(f"Data processing error: {str(e)}")
    
    return None

def calculate_indicators(df):
    """Calculate technical indicators with enhanced features"""
    if df is None or df.empty or len(df) < 20:
        app.logger.error("Insufficient data for indicators")
        return None

    try:
        # Original indicators
        df["rsi"] = ta.rsi(df["close"], length=14)
        bbands = ta.bbands(df["close"], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)
        df = df.rename(columns={
            "BBU_20_2.0": "upper_band",
            "BBL_20_2.0": "lower_band"
        }).dropna()
        
        # New indicators
        df["ema_50"] = ta.ema(df["close"], length=50)
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df = pd.concat([df, adx], axis=1)
        
        # Bollinger Band width analysis
        df["band_width"] = df["upper_band"] - df["lower_band"]
        df["band_width_ma"] = df["band_width"].rolling(20).mean()

        required_cols = ["close", "rsi", "upper_band", "lower_band",
                        "ema_50", "ADX_14", "band_width", "band_width_ma"]
        if not all(col in df.columns for col in required_cols):
            app.logger.error("Missing required columns after calculations")
            return None

        return df.iloc[-100:]

    except Exception as e:
        app.logger.error(f"Indicator error: {str(e)}")
        return None

def generate_signals(df):
    """Generate trading signals with enhanced conditions"""
    if df is None or df.empty:
        return []
    
    try:
        signals = []
        latest = df.iloc[-1]
        prev = df.iloc[-5]  # 5 periods back for momentum check

        # Original signals
        if latest["rsi"] > 70:
            signals.append("RSI Overbought")
        elif latest["rsi"] < 30:
            signals.append("RSI Oversold")
            
        if latest["close"] > latest["upper_band"]:
            signals.append("Price Above Upper Band")
        elif latest["close"] < latest["lower_band"]:
            signals.append("Price Below Lower Band")

        # New signals
        # 1. Bollinger Band Squeeze
        if latest["band_width"] < 0.5 * latest["band_width_ma"]:
            signals.append("Volatility Squeeze Detected")

        # 2. Momentum Check
        if latest["rsi"] > 50 and latest["close"] > prev["close"]:
            signals.append("Bullish Momentum")
        elif latest["rsi"] < 50 and latest["close"] < prev["close"]:
            signals.append("Bearish Momentum")

        # 3. EMA Trend
        if latest["close"] > latest["ema_50"]:
            signals.append("Price Above EMA50")
        else:
            signals.append("Price Below EMA50")

        # 4. ADX Trend Strength
        if latest["ADX_14"] > 25:
            if latest["close"] > prev["close"]:
                signals.append("Strong Uptrend (ADX > 25)")
            else:
                signals.append("Strong Downtrend (ADX > 25)")

        return signals
    
    except Exception as e:
        app.logger.error(f"Signal generation error: {str(e)}")
        return []

@app.route("/")
def main_endpoint():
    try:
        df = get_futures_kline()
        if df is None:
            return jsonify({
                "status": "error",
                "message": "Failed to fetch market data",
                "timestamp": datetime.utcnow().isoformat()
            }), 500

        processed_df = calculate_indicators(df)
        if processed_df is None:
            return jsonify({
                "status": "error",
                "message": "Failed to calculate indicators",
                "data_points": len(df)
            }), 500

        signals = generate_signals(processed_df)
        latest = processed_df.iloc[-1]

        return jsonify({
            "status": "success",
            "signals": signals,
            "data": {
                "price": round(latest["close"], 2),
                "rsi": round(latest["rsi"], 2),
                "upper_band": round(latest["upper_band"], 2),
                "lower_band": round(latest["lower_band"], 2),
                "ema_50": round(latest["ema_50"], 2),
                "adx": round(latest["ADX_14"], 2),
                "timestamp": latest["timestamp"].isoformat()
            },
            "meta": {
                "symbol": DEFAULT_SYMBOL,
                "interval": DEFAULT_INTERVAL,
                "candles_analyzed": len(processed_df)
            }
        })

    except Exception as e:
        app.logger.critical(f"Critical error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
