import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "^GSPC_stock_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "^GSPC_stock_data_processed.csv")

def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df.dropna(subset=["Close", "Open"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Calculate gap: today's Open minus previous day's Close
    df["Prev_Close"] = df["Close"].shift(1)
    df["Gap"] = df["Open"] - df["Prev_Close"]
    
    df.dropna(inplace=True)  # drop first row due to shift
    
    return df

def create_features(df, lags=5, rolling_windows=[3, 7]):
    features = ["Open", "Close", "Gap"]
    for feat in features:
        for lag in range(1, lags + 1):
            df[f"{feat}_lag_{lag}"] = df[feat].shift(lag)
        for window in rolling_windows:
            df[f"{feat}_roll_mean_{window}"] = df[feat].rolling(window=window).mean()
            df[f"{feat}_roll_std_{window}"] = df[feat].rolling(window=window).std()
    
    # Shift targets by -1 to predict next day's values
    df["Open_target"] = df["Open"].shift(-1)
    df["Close_target"] = df["Close"].shift(-1)
    df["Gap_target"] = df["Gap"].shift(-1)
    
    df.dropna(inplace=True)
    return df

def split_data(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    return train_df, test_df

def save_processed_data(df, path=PROCESSED_DATA_PATH):
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")

if __name__ == "__main__":
    df = load_raw_data()
    df = create_features(df)
    train_df, test_df = split_data(df)
    save_processed_data(df)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
