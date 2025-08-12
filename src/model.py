# src/model.py
import os
import torch
import joblib
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.features import add_features, add_features_multi_asset
from src.data_fetch import fetch_bars, fetch_multiple_symbols
from src.utils import NormalizedTimeSeriesDataset, plot_predictions, plot_residuals

class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, seq_len=30):
        self.seq_len = seq_len
        self.X = []
        self.y = []
        values = df[feature_cols].values
        targets = df["target_return"].values
        for i in range(len(df) - seq_len):
            self.X.append(values[i:i+seq_len])
            self.y.append(targets[i+seq_len-1])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=6, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # single continuous output
        )

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer_encoder(x)
        out = out[:, -1, :]  # last time step
        return self.reg_head(out)

def train_and_save(symbol="AAPL", start="2020-01-01", end="2025-01-01", model_out="models/ts_transformer.pt"):
    symbols = ["AAPL", "MSFT", "SPY", "NVDA"]  # your main symbol + others

    data = fetch_multiple_symbols(symbols, start, end)
    main_df = data["AAPL"]
    other_dfs = {sym: df for sym, df in data.items() if sym != "AAPL"}

    df, feature_cols = add_features_multi_asset(main_df, other_dfs)
    
    #df = fetch_bars(symbol, start=start, end=end)
    #df, feature_cols = add_features(df)

    seq_len = 30
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_ds = NormalizedTimeSeriesDataset(df.iloc[:train_end], feature_cols, fit_scalers=True)
    val_ds = NormalizedTimeSeriesDataset(df.iloc[train_end:val_end], feature_cols,
                                        feature_scaler=train_ds.feature_scaler,
                                        target_scaler=train_ds.target_scaler)
    test_ds = NormalizedTimeSeriesDataset(df.iloc[val_end:], feature_cols,
                                         feature_scaler=train_ds.feature_scaler,
                                         target_scaler=train_ds.target_scaler)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTimeSeries(len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_epoch = -1
    best_model_state = None

    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

                all_val_preds.extend(preds.cpu().numpy().flatten())
                all_val_true.extend(y_batch.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    all_test_preds = []
    all_test_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            all_test_preds.extend(preds.cpu().numpy().flatten())
            all_test_true.extend(y_batch.cpu().numpy().flatten())

    mse = mean_squared_error(all_test_true, all_test_preds)
    mae = mean_absolute_error(all_test_true, all_test_preds)
    r2 = r2_score(all_test_true, all_test_preds)
    print(f"\nBest epoch {best_epoch} with val loss {best_val_loss:.4f}")
    print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"Saved best regression model to {model_out}")

    joblib.dump(train_ds.feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(train_ds.target_scaler, "models/target_scaler.pkl")

    plot_predictions(all_test_true, all_test_preds)
    plot_residuals(all_test_true, all_test_preds)


if __name__ == "__main__":
    train_and_save()
