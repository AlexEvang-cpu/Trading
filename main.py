import os
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, jsonify

# Configuration
MEXC_FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"
DEFAULT_SYMBOL = os.getenv("TRADING_SYMBOL", "BTC_USDT")
DEFAULT_INTERVAL = os.getenv("TRADING_INTERVAL", "Min15")
DEFAULT_LIMIT = int(os.getenv("DATA_LIMIT", "100"))

app = Flask(__name__)

def get_futures_kline(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=DEFAULT_LIMIT):
    """Safely fetch futures market data with enhanced error handling"""
    endpoint = f"{MEXC_FUTURES_BASE_URL}/kline/{symbol}"
    params = {"interval": interval, "limit": limit}
    
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if not data.get("success", False):
            app.logger.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None

        raw_data = data.get("data", [])
        if not raw_data or len(raw_data) < 10:  # Ensure minimum data points
            app.logger.warning("Insufficient data from API")
            return None

        # Validate and structure data
        columns = ["timestamp", "open", "high", "low", "close", "volume", "amount", "trade_count"]
        df = pd.DataFrame(raw_data, columns=columns)
        
        # Type conversions
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        
        # Clean data
        df = df.dropna()
        if df.empty:
            app.logger.error("Empty DataFrame after cleaning")
            return None
            
        return df

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error: {str(e)}")
    except Exception as e:
        app.logger.error(f"Data processing error: {str(e)}")
    
    return None

def calculate_indicators(df):
    """Safe indicator calculation with validation"""
    if df is None or df.empty or len(df) < 20:
        app.logger.error("Insufficient data for indicators")
        return None

    try:
        # Calculate RSI
        df["rsi"] = ta.rsi(df["close"], length=14)
        
        # Calculate Bollinger Bands
        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is None or bbands.empty:
            app.logger.error("Bollinger Bands calculation failed")
            return None
            
        df = pd.concat([df, bbands], axis=1)
        df.rename(columns={
            "BBU_20_2.0": "upper_band",
            "BBL_20_2.0": "lower_band"
        }, inplace=True)

        # Post-calculation validation
        required_columns = ["close", "rsi", "upper_band", "lower_band"]
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            app.logger.error(f"Missing columns: {missing}")
            return None

        # Remove rows with invalid values
        df = df[(df["rsi"].between(0, 100)) & 
               (df["upper_band"] > df["lower_band"])].dropna()
        
        return df[-100:]  # Return most recent 100 data points

    except Exception as e:
        app.logger.error(f"Indicator error: {str(e)}")
        return None

def generate_signals(df):
    """Generate trading signals with safety checks"""
    if df is None or df.empty:
        return []
    
    try:
        latest = df.iloc[-1]
        signals = []

        # RSI Signals
        if latest["rsi"] > 70:
            signals.append("RSI Overbought (Potential Sell)")
        elif latest["rsi"] < 30:
            signals.append("RSI Oversold (Potential Buy)")

        # Bollinger Bands Signals
        if latest["close"] > latest["upper_band"]:
            signals.append("Price Above Upper Band (Possible Short)")
        elif latest["close"] < latest["lower_band"]:
            signals.append("Price Below Lower Band (Possible Long)")

        return signals
    
    except IndexError:
        app.logger.error("Index error in signal generation")
    except KeyError as e:
        app.logger.error(f"Missing data column: {str(e)}")
    
    return []

@app.route("/")
def main_handler():
    """Main trading signal endpoint"""
    try:
        # Step 1: Fetch data
        df = get_futures_kline()
        if df is None or df.empty:
            return jsonify({
                "status": "error",
                "message": "Data fetch failed",
                "advice": [
                    "Check symbol/interval parameters",
                    "Verify API availability",
                    "Ensure network connectivity"
                ]
            }), 500

        # Step 2: Calculate indicators
        processed_df = calculate_indicators(df)
        if processed_df is None or processed_df.empty:
            return jsonify({
                "status": "error",
                "message": "Indicator calculation failed",
                "advice": [
                    "Try increasing DATA_LIMIT",
                    "Check data quality from /debug endpoint"
                ]
            }), 500

        # Step 3: Generate signals
        signals = generate_signals(processed_df)
        
        # Prepare response
        response = {
            "status": "success",
            "signals": signals,
            "meta": {
                "symbol": DEFAULT_SYMBOL,
                "interval": DEFAULT_INTERVAL,
                "data_points": len(processed_df)
            }
        }

        # Add latest data if available
        try:
            latest = processed_df.iloc[-1]
            response["latest"] = {
                "price": round(latest["close"], 2),
                "rsi": round(latest["rsi"], 2),
                "upper_band": round(latest["upper_band"], 2),
                "lower_band": round(latest["lower_band"], 2)
            }
        except Exception as e:
            app.logger.error(f"Data packaging error: {str(e)}")

        return jsonify(response)

    except Exception as e:
        app.logger.critical(f"Critical failure: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "advice": "Check application logs for details"
        }), 500

@app.route("/debug")
def debug_endpoint():
    """Debugging endpoint for data inspection"""
    df = get_futures_kline(limit=5)
    
    if df is None or df.empty:
        return jsonify({"status": "error", "message": "No data available"}), 500
        
    return jsonify({
        "status": "success",
        "data_columns": list(df.columns),
        "sample_data": df.iloc[0].to_dict(),
        "data_stats": {
            "start_time": df["timestamp"].min(),
            "end_time": df["timestamp"].max(),
            "rows": len(df),
            "na_values": df.isna().sum().to_dict()
        }
    })

@app.route("/health")
def health_check():
    """Basic health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": pd.Timestamp.now().isoformat()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
