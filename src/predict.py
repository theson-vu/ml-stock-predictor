# src/predict.py
import torch
import pandas as pd
import joblib
import numpy as np
from torch.utils.data import DataLoader
from src.data_fetch import fetch_multiple_symbols
from src.features import add_features_multi_asset
from src.model import TransformerTimeSeries, TimeSeriesDataset
import matplotlib.pyplot as plt
from src.utils import NormalizedTimeSeriesDataset

def infer(symbol="AAPL", model_path="models/ts_transformer.pt", start="2023-01-01", end="2024-01-01"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    symbols = [symbol, "MSFT", "SPY", "NVDA"]  # same symbols as training

    # Fetch data for all symbols
    data = fetch_multiple_symbols(symbols, start=start, end=end)
    main_df = data[symbol]
    other_dfs = {sym: df for sym, df in data.items() if sym != symbol}

    # Add multi-asset features
    df, feature_cols = add_features_multi_asset(main_df, other_dfs)

    feature_scaler = joblib.load("models/feature_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    seq_len = 30
    dataset = NormalizedTimeSeriesDataset(df, feature_cols,
                                          feature_scaler=feature_scaler,
                                          target_scaler=target_scaler,
                                          fit_scalers=False)
    loader = DataLoader(dataset, batch_size=32)

    # Load model
    model = TransformerTimeSeries(len(feature_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).squeeze(1)  # output shape [batch]
            preds_scaled.extend(pred.cpu().numpy())

    # Inverse scale predictions
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = target_scaler.inverse_transform(preds_scaled).flatten()

    df = df.iloc[seq_len:].copy()
    df["pred"] = preds
    return df.tail(20)

if __name__ == "__main__":
    print(infer())
