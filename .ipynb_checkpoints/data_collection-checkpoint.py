import yfinance as yf
import pandas as pd
import os

DATA_DIR = "data"
TICKER = "^GSPC"  # S&P 500 Index ticker on Yahoo Finance
PERIOD = "1y"
CSV_PATH = os.path.join(DATA_DIR, f"{TICKER}_stock_data.csv")

def fetch_stock_data(ticker=TICKER, period=PERIOD):
    print(f"Fetching data for {ticker} over period: {period}")
    df = yf.download(ticker, period=period)[["Open", "Close"]].reset_index()
    df.dropna(inplace=True)
    return df

def save_data(df, path=CSV_PATH):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    df = fetch_stock_data()
    save_data(df)
