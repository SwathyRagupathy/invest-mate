# visualize_prophet_forecasts.py

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

DATA_PATH = "data/^GSPC_stock_data_processed.csv"

def plot_prophet(target_col, title):
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    prophet_df = df[["Date", target_col]].rename(columns={"Date": "ds", target_col: "y"})

    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df.iloc[:-1])  # train on all but last row

    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    # Plot
    fig = model.plot(forecast)
    plt.title(f"{title}: Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_prophet("Open_target", "Next-Day Open Price")
    plot_prophet("Close_target", "Next-Day Close Price")
    plot_prophet("Gap_target", "Next-Day Gap (Open - Prev Close)")
