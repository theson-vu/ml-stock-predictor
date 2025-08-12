import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # MACD
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-day)
    df["bb_mid"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # ATR (14-day)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv

    return df


def add_features_multi_asset(main_df, other_dfs: dict):
    """
    Adds technical indicators for the main_df plus related assets.
    other_dfs: dict of {symbol: DataFrame} for other stocks/indexes.
    All dataframes should have datetime index aligned.
    Returns augmented DataFrame and list of new feature columns.
    """

    df = main_df.copy()

    # Your existing features on main_df
    df["return"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["volatility_5"] = df["return"].rolling(5).std()

    # RSI as before
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Now add features from other assets
    for sym, other_df in other_dfs.items():
        # Align indexes, reindex to main df index if needed
        other_df = other_df.reindex(df.index).fillna(method='ffill')

        # Example: just add close returns and RSI of related asset
        df[f"{sym}_return"] = other_df["close"].pct_change()
        
        # RSI for related asset
        delta_o = other_df["close"].diff()
        gain_o = delta_o.clip(lower=0)
        loss_o = -delta_o.clip(upper=0)
        avg_gain_o = gain_o.rolling(14).mean()
        avg_loss_o = loss_o.rolling(14).mean()
        rs_o = avg_gain_o / (avg_loss_o + 1e-9)
        df[f"{sym}_rsi"] = 100 - (100 / (1 + rs_o))

    # Target
    df["target_return"] = df["close"].shift(-1) / df["close"] - 1
    #df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna()
    feature_cols = [c for c in df.columns if c not in ["target"]]

    return df, feature_cols


def add_features(df: pd.DataFrame, rsi_period: int = 14):
    df = df.copy()
    df = compute_technical_indicators(df)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Returns & volatility
    df["return"] = df["close"].pct_change()
    df["volatility_5"] = df["return"].rolling(5).std()

    # Target: next-day up/down
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop NaNs resulting from indicators
    df.dropna(inplace=True)

    # Features to scale
    exclude_cols = ["target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]

    # Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols
