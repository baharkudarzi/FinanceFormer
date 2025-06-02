# scripts/build_dataset.py

import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler

def compute_indicators(df):
    # ATR(14)
    high_low = df["High"] - df["Low"]
    high_prevclose = (df["High"] - df["Close"].shift(1)).abs()
    low_prevclose  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prevclose, low_prevclose], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD & Signal
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df

def build_dataset(
    csv_path="data/volatility_labeled.csv",
    seq_len=60,
    pred_horizon=5,
    save_path="data/dataset.pt"
):
    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 2) Coerce numeric & drop bad rows
    for col in ["Open","High","Low","Close","Volume","LogReturn","Volatility_21d"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open","High","Low","Close","Volume","LogReturn","Volatility_21d"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3) Compute new indicators
    df = compute_indicators(df)
    # Drop initial NaNs from indicator windows
    df.dropna(subset=["ATR_14","RSI_14","MACD","MACD_Signal"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 4) Define features + target
    features = [
        "Open","High","Low","Close","Volume","LogReturn",
        "ATR_14","RSI_14","MACD","MACD_Signal"
    ]
    target_col = "Volatility_21d"

    # 5) Scale features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # 6) Build sliding windows
    X_windows, y_targets = [], []
    N = len(df)
    for i in range(N - seq_len - pred_horizon + 1):
        X_windows.append(df[features].iloc[i : i + seq_len].values)
        y_targets.append(df[target_col].iloc[i + seq_len + pred_horizon - 1])

    X = torch.tensor(np.stack(X_windows), dtype=torch.float32)
    y = torch.tensor(np.array(y_targets), dtype=torch.float32)

    # 7) Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save((X, y), save_path)
    print(f"Saved dataset to {save_path}")
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

if __name__ == "__main__":
    build_dataset()
