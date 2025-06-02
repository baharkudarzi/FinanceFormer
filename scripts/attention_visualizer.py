# scripts/attention_visualizer.py

import torch
import matplotlib.pyplot as plt
from models.transformer_model import TransformerRegressor

def extract_attention_weights(model, x):
    """
    Runs a forward pass through the first encoder layer,
    returns its self-attention weights for input x,
    averaged over all heads.
    """
    # x: (B, T, F)
    # 1) Project & add pos encoding
    x_proj = model.input_proj(x)
    x_pe   = model.pos_encoder(x_proj)
    # 2) Transpose for (T, B, D)
    x_enc = x_pe.transpose(0, 1)
    # 3) First encoder layer
    layer0 = model.transformer.layers[0]
    # 4) Get raw attention weights (B, num_heads, T, T)
    _, attn_weights = layer0.self_attn(
        x_enc, x_enc, x_enc,
        need_weights=True,
        average_attn_weights=False
    )
    # 5) Average over heads → (B, T, T)
    attn_weights = attn_weights.mean(dim=1)
    return attn_weights.detach().cpu().numpy()

def main(
    data_path="data/dataset.pt",
    model_path="models/best_model.pt",
    num_samples=3
):
    # Load test windows
    X, _ = torch.load(data_path)
    total = X.shape[0]
    train_size = int(0.7 * total)
    val_size   = int(0.15 * total)
    start_test = train_size + val_size
    X_test = X[start_test:]  # (N_test, T, F)

    # Load model
    model = TransformerRegressor(input_dim=X.shape[2])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Plot attention for first few samples
    for i in range(min(num_samples, len(X_test))):
        window = X_test[i].unsqueeze(0)  # (1, T, F)
        attn = extract_attention_weights(model, window)[0]  # (T, T)

        plt.figure(figsize=(6, 5))
        plt.imshow(attn, aspect='auto')
        plt.colorbar(label="Attention weight")
        plt.title(f"Sample {i} — Layer0 Self-Attention")
        plt.xlabel("Key position (t)")
        plt.ylabel("Query position (t')")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
