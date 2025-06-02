# scripts/fetch_and_label_volatility.py

import yfinance as yf
import pandas as pd
import numpy as np
import os

def compute_rolling_volatility(close_prices, window=21):
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.rolling(window=window).std()

def fetch_and_label(ticker, start_date="2015-01-01", end_date="2024-12-31", save_path="data/volatility_labeled.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = yf.download(ticker, start=start_date, end=end_date)
    df.dropna(inplace=True)

    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility_21d"] = df["LogReturn"].rolling(window=21).std() * np.sqrt(252)  # annualized

    df.dropna(inplace=True)
    df.to_csv(save_path)
    print(f"Saved labeled data with volatility to {save_path}")

if __name__ == "__main__":
    fetch_and_label("AAPL")
