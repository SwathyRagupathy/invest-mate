# invest-mate📈

# 📌 Objective

To build a production-quality time series forecasting system using stock price data, comparing four key modeling approaches—Prophet, XGBoost, LSTM, and N-BEATS—with a full pipeline including data scraping, preprocessing, model evaluation, interactive dashboarding, and automatic updates every 30 minutes.

# 🔁 Workflow

1. Data Collection (via yFinance API)
2. Data Preprocessing (lag features, normalization)
3. Modeling:
   - Prophet (classical)
   - XGBoost (ML)
   - LSTM (DL)
   - N-BEATS (SOTA DL)
4. Model Evaluation (RMSE, MAPE, MAE, R²)
5. Visualization:
   - Actual vs Forecasted Plots
   - Dashboard with model comparison & metrics
6. Auto-Update Forecast:
   - Every 30 minutes
   - Prophet model retrains with latest data
   - New forecast plotted on dashboard


# Latest

 From classical models like ARIMA and Prophet to LSTMs, time series forecasting has come a long way. But today, I explored some of the latest innovations that are pushing the boundaries:

🔍 1. PatchTST
A Transformer model that breaks the time series into non-sequential patches, allowing it to capture long-range patterns more efficiently than RNNs or vanilla Transformers.

🧩 2. Temporal Fusion Transformers (TFT)
Combines static and dynamic features with attention layers and interpretable outputs. It’s especially useful for complex multivariate forecasting problems in domains like finance, energy, and retail.

⚡ 3. N-BEATS & N-HITS
Deep learning models designed exclusively for forecasting — they work well even without handcrafted features. Their robustness and performance have made them strong SOTA contenders.

<img width="679" alt="Screenshot 2025-07-04 at 8 37 48 PM" src="https://github.com/user-attachments/assets/f0bc2a03-b3c1-420f-9aa5-95565ee95355" />

