# scripts/evaluate.py

import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.transformer_model import TransformerRegressor
import numpy as np

def evaluate(
    data_path="data/dataset.pt",
    model_path="models/best_model.pt"
):
    # Load data
    X, y = torch.load(data_path)
    total = X.shape[0]
    train_size = int(0.7 * total)
    val_size   = int(0.15 * total)
    start_test = train_size + val_size

    X_test = X[start_test:].numpy()
    y_true = y[start_test:].numpy()

    # Instantiate with the same hyperparams you trained with
    input_dim = X.shape[2]
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=1
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae  = mean_absolute_error(y_true, preds)
    r2   = r2_score(y_true, preds)

    print("Test Set Metrics:")
    print(f"  • RMSE: {rmse:.4f}")
    print(f"  • MAE:  {mae:.4f}")
    print(f"  • R²:   {r2:.4f}")

if __name__ == "__main__":
    evaluate()
