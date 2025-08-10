# src/features.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["volatility_5"] = df["return"].rolling(5).std()

    # RSI: standard implementation (no lookahead)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Target: next-day up (1) / down (0)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()
