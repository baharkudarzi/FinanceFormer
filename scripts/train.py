import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.transformer_model import TransformerRegressor
import os

def train_model(
    data_path="data/dataset.pt",
    model_save_path="models/best_model.pt",
    epochs=20,
    batch_size=32,
    lr=1e-3
):
    # Load dataset
    X, y = torch.load(data_path)
    print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")

    dataset = TensorDataset(X, y)

    # Split: 70% train, 15% val, 15% test
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Initialize model
    input_dim = X.shape[2]
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_layers=1
    ).to("cpu")


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else float("nan")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float("nan")

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  🔥 Best model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
