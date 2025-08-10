# src/model.py
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import pandas as pd
from src.features import add_features
from src.data_fetch import fetch_bars

def train_and_save(symbol="AAPL", start="2020-01-01", end="2023-01-01", model_out="models/rf.joblib"):
    df = fetch_bars(symbol, start=start, end=end)
    df = add_features(df)
    X = df[["sma_5","sma_10","ema_10","return","volatility_5","rsi"]]
    y = df["target"]

    # time-aware train/test
    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # Make sure output folder exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    train_and_save()
