import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def prepare_training_data(symbol="BTC_USDT", interval="Min1", limit=5000):
    # Fetch historical data
    df = get_futures_kline(symbol, interval, limit)
    df = calculate_indicators(df)
    
    # Create labels (1 if next candle is green, 0 otherwise)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    # Features
    features = df[["rsi", "close", "volatility", "MACD_12_26_9", "volume_change"]]
    
    # Remove NaNs
    valid_idx = features.dropna().index
    return features.loc[valid_idx], df["target"].loc[valid_idx]

def train_model():
    X, y = prepare_training_data()
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X, y)
    dump(model, "model.joblib")
    print("Model trained and saved")

if __name__ == "__main__":
    train_model()
