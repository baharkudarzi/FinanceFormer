# scripts/hyperparameter_search.py

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.transformer_model import TransformerRegressor
import itertools
import os

def evaluate_params(params, X, y, epochs=5, batch_size=32, lr=1e-3):
    # Split into train/val
    total = X.shape[0]
    train_sz = int(0.7 * total)
    val_sz   = int(0.15 * total)
    train_set, val_set, _ = random_split(
        TensorDataset(X, y),
        [train_sz, val_sz, total - train_sz - val_sz]
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    # Model
    model = TransformerRegressor(
        input_dim=X.shape[2],
        d_model   = params["d_model"],
        nhead     = params["nhead"],
        num_layers= params["num_layers"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        # validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
    return best_val

def main():
    # 1) Load data once
    X, y = torch.load("data/dataset.pt")

    # 2) Define grid
    grid = {
        "d_model":    [32, 64, 128],
        "nhead":      [2, 4],
        "num_layers": [1, 2],
    }
    combos = list(itertools.product(
        grid["d_model"], grid["nhead"], grid["num_layers"]
    ))

    results = []
    for d_model, nhead, num_layers in combos:
        params = {"d_model": d_model, "nhead": nhead, "num_layers": num_layers}
        print(f"Testing {params} ...", end=" ")
        val_loss = evaluate_params(params, X, y)
        print(f"Val Loss: {val_loss:.4f}")
        results.append((params, val_loss))

    # 3) Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/hyperparam_search.txt", "w") as f:
        for params, loss in results:
            f.write(f"{params} -> {loss:.4f}\n")
    print("\nSearch complete. See results/hyperparam_search.txt")

if __name__ == "__main__":
    main()
