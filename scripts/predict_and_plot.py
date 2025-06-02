# scripts/predict_and_plot.py

import torch
from models.transformer_model import TransformerRegressor
import matplotlib.pyplot as plt
import os

def predict_and_plot(
    data_path="data/dataset.pt",
    model_path="models/best_model.pt"
):
    # 1) Load dataset
    X, y = torch.load(data_path)
    total = X.shape[0]
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    # 2) Extract test split
    start_test = train_size + val_size
    X_test = X[start_test:]
    y_test = y[start_test:]

    # 3) Load model
    input_dim = X.shape[2]
    model = TransformerRegressor(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 4) Predict
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
    y_true = y_test.cpu().numpy()

    # 5) Plot
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(preds, label="Predicted")
    plt.title("Volatility Forecast: True vs Predicted")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.tight_layout()

    # 6) Show or save
    plt.show()

if __name__ == "__main__":
    predict_and_plot()
