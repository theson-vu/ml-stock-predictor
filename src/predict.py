# src/predict.py
import joblib, pandas as pd
from src.features import add_features
from src.data_fetch import fetch_bars

def infer(symbol, model_path="models/rf.joblib"):
    model = joblib.load(model_path)
    df = fetch_bars(symbol, start="2023-01-01", end="2024-01-01")  # example range
    df = add_features(df)
    X = df[["sma_5","sma_10","ema_10","return","volatility_5","rsi"]]
    probs = model.predict_proba(X)[:,1]
    df["pred_proba_up"] = probs
    return df.tail(10)

if __name__ == "__main__":
    print(infer("AAPL"))
