# src/utils.py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class NormalizedTimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols, target_col="target_return", seq_len=30,
                 feature_scaler=None, target_scaler=None, fit_scalers=False):
        self.seq_len = seq_len

        # Extract raw feature and target values
        features = df[feature_cols].values
        targets = df[target_col].values.reshape(-1, 1)

        # Fit scalers on training data only
        if fit_scalers:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            self.feature_scaler.fit(features)
            self.target_scaler.fit(targets)
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

        # Transform features and targets
        features = self.feature_scaler.transform(features)
        targets = self.target_scaler.transform(targets).flatten()

        # Create sequences
        self.X = []
        self.y = []
        for i in range(len(df) - seq_len):
            self.X.append(features[i:i+seq_len])
            self.y.append(targets[i+seq_len-1])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)  # regression target as float

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


def plot_predictions(true_vals, pred_vals, title="Predicted vs True Returns"):
    plt.figure(figsize=(10, 6))
    plt.plot(true_vals, label='True Returns', alpha=0.7)
    plt.plot(pred_vals, label='Predicted Returns', alpha=0.7)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Returns")
    plt.legend()
    plt.show()

def plot_residuals(true_vals, pred_vals, title="Residual Errors"):
    residuals = np.array(true_vals) - np.array(pred_vals)
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()
