# prophet_model.py

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import os

# === Configuration ===
DATA_DIR = "data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "^GSPC_stock_data_processed.csv")
FORECASTS_DIR = os.path.join(DATA_DIR, "prophet_forecasts")
os.makedirs(FORECASTS_DIR, exist_ok=True)

# === Prepare Prophet Input ===
def prepare_prophet_df(df, target_col):
    # Prepare dataframe for Prophet: columns must be ['ds', 'y']
    prophet_df = df[["Date", target_col]].rename(columns={"Date": "ds", target_col: "y"})
    return prophet_df

# === Train Prophet and Forecast Next Day ===
def train_and_forecast(prophet_df, target):
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    # Forecast for the next day
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    # Extract and format next-day forecast
    forecast_next = forecast.tail(1)[["ds", "yhat"]].rename(
        columns={"ds": "predicted_date", "yhat": "predicted_value"}
    )
    forecast_next["target"] = target

    return model, forecast_next

# === Metrics ===
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mape

# === Main Execution ===
if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])

    targets = ["Open_target", "Close_target", "Gap_target"]

    results = []
    forecasts = []

    for target in targets:
        print(f"\n📊 Training Prophet model for {target}...")

        prophet_df = prepare_prophet_df(df, target)

        # Use all but last row for training, last row for test
        train_df = prophet_df.iloc[:-1]
        test_df = prophet_df.iloc[-1:]

        model, forecast_next = train_and_forecast(train_df, target)

        y_true = test_df["y"].values
        y_pred = forecast_next["predicted_value"].values

        rmse, mape = evaluate(y_true, y_pred)
        print(f"{target} - ✅ RMSE: {rmse:.4f}, ✅ MAPE: {mape:.2f}%")

        results.append({
            "target": target,
            "rmse": rmse,
            "mape": mape,
            "predicted_date": forecast_next["predicted_date"].values[0],
            "predicted_value": y_pred[0]
        })

        forecasts.append(forecast_next)

    # Save forecasts to CSV
    all_forecasts = pd.concat(forecasts, ignore_index=True)
    forecast_csv_path = os.path.join(FORECASTS_DIR, "next_day_forecasts.csv")
    all_forecasts.to_csv(forecast_csv_path, index=False)
    print(f"\n📁 Saved next day forecasts to: {forecast_csv_path}")

    # Summary
    results_df = pd.DataFrame(results)
    print("\n📈 Evaluation Summary:")
    print(results_df)
