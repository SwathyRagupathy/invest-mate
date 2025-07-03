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



