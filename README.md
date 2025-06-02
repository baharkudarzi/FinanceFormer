# FinanceFormer

**Volatility Forecasting with Transformers**

---

## 🔍 Overview

**FinanceFormer** uses a Transformer‐based encoder to predict future **annualized volatility** of a single financial asset (AAPL by default) from its historical OHLCV time series + technical indicators (ATR, RSI, MACD).  

Key ideas:
- Sequence-to-one forecasting: input = last 60 days of features, output = t+5 volatility  
- Attention mechanism highlights which past days matter most  
- Achieves **R² ≈ 0.74** on held-out test data  

---

## 📦 Installation

```bash
# create & activate venv
python3 -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt
```

**requirements.txt** should include:
```
yfinance
pandas
numpy
torch
scikit-learn
matplotlib
```

---

## 🚀 Usage

Run in this order from project root:

1. **Fetch & label data**  
   ```bash
   python -m scripts.fetch_and_label_volatility
   ```
2. **Build dataset**  
   ```bash
   python -m scripts.build_dataset
   ```
3. **Train model**  
   ```bash
   python -m scripts.train
   ```
4. **Evaluate performance**  
   ```bash
   python -m scripts.evaluate
   # ➞ RMSE: 0.0270 • MAE: 0.0207 • R²: 0.7370
   ```
5. **Visualize attention**  
   ```bash
   python -m scripts.attention_visualizer
   ```
6. **Plot predictions**  
   ```bash
   python -m scripts.predict_and_plot
   ```

---

## 📊 Results

- **Test RMSE:** 0.0270  
- **Test MAE:**  0.0207  
- **Test R²:**   0.7370  

<p align="center">
  <img src="results/pred_vs_true.png" alt="Predicted vs True Volatility" width="600"/>
</p>
<p align="center">
  <img src="results/attention_heatmap_sample0.png" alt="Attention Heatmap" width="600"/>
</p>

---

## 🔧 Hyperparameters

- **seq_len:** 60  
- **pred_horizon:** 5  
- **d_model:** 128  
- **nhead:** 4  
- **num_layers:** 1  
- **Batch size:** 32  
- **Learning rate:** 1e-3  
- **Epochs:** 20  

---

## 🚀 Next Steps

- Multi‐horizon forecasting (t+1, t+3, t+5)  
- Add macro or sentiment features  
- Deploy as a REST API for live volatility predictions  

---

## 🔗 License & Citation

Feel free to fork or cite this repository in your work.  
*By Bahareh Kudarzi*
