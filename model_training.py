import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

def fetch_training_data(symbol="BTC_USDT", interval="Min1", limit=5000):
    """Fetch and preprocess historical data for training"""
    print("🔄 Fetching historical data...")
    data = get_futures_kline(symbol, interval, limit)
    if data is None:
        raise ValueError("Failed to fetch training data")
        
    print("🧮 Calculating indicators...")
    processed = calculate_indicators(data)
    if processed is None:
        raise ValueError("Indicator calculation failed")
    
    return processed

def create_features_target(df):
    """Engineer features and target variable"""
    print("🎯 Creating features and target...")
    
    # Target: Next candle direction (1=up, 0=down)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    # Feature set
    features = df[[
        'rsi', 'ema_20', 'bb_BBU_14_1.5', 'bb_BBL_14_1.5',
        'volume_ma', 'volume_spike', 'hour'
    ]]
    
    # Lag features
    for lag in [1, 3, 5]:
        features[f"rsi_lag{lag}"] = df["rsi"].shift(lag)
        features[f"volume_lag{lag}"] = df["volume"].shift(lag)
    
    # Clean data
    valid_idx = features.dropna().index
    return features.loc[valid_idx], df["target"].loc[valid_idx]

def train_evaluate_model(X, y):
    """Train and validate model with hyperparameter tuning"""
    print("🤖 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    
    # Grid search with cross-validation
    model = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"✅ Best params: {model.best_params_}")
    print(f"🏆 Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred))
    
    # Plot feature importance
    plt.figure(figsize=(10,6))
    feat_importances = pd.Series(model.best_estimator_.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Important Features")
    plt.show()
    
    return model.best_estimator_

def save_model(model, path="price_model.joblib"):
    """Save trained model to disk"""
    print(f"💾 Saving model to {path}...")
    dump(model, path)
    print("Model saved successfully")

# Full training pipeline
try:
    # Step 1: Get Data
    df = fetch_training_data()
    
    # Step 2: Feature Engineering
    X, y = create_features_target(df)
    print(f"📊 Training data shape: {X.shape}")
    
    # Step 3: Model Training
    best_model = train_evaluate_model(X, y)
    
    # Step 4: Save Model
    save_model(best_model)
    
except Exception as e:
    print(f"❌ Training failed: {str(e)}")
    raise
