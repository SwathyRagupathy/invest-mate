# xgboost_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import os
import joblib

DATA_DIR = "data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "^GSPC_stock_data_processed.csv")
MODEL_DIR = os.path.join("models", "xgboost")
os.makedirs(MODEL_DIR, exist_ok=True)

# === Feature Engineering ===
def create_features(df, target_col, lags=[1,2,3], windows=[3,7,14]):
    df = df.copy()
    
    # Lag features
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    # Rolling mean/std features
    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = df[target_col].shift(1).rolling(window=window).mean()
        df[f"{target_col}_roll_std_{window}"] = df[target_col].shift(1).rolling(window=window).std()
    
    # Date/time features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    
    # Drop rows with NaN (due to lagging)
    df = df.dropna().reset_index(drop=True)
    return df

# === Train/Test Split ===
def train_test_split_time_series(df, target_col, test_size=0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    X_train = train.drop(columns=[target_col, "Date"])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col, "Date"])
    y_test = test[target_col]
    return X_train, X_test, y_train, y_test

# === Train and Evaluate ===
def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=1
    )
    model.fit(X_train, y_train)  # No early stopping
    
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100
    return model, preds, rmse, mape

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])

    # Targets to model
    targets = ["Open_target", "Close_target", "Gap_target"]

    results = []
    models = {}

    for target in targets:
        print(f"\n🔎 Processing target: {target}")

        df_fe = create_features(df, target)
        X_train, X_test, y_train, y_test = train_test_split_time_series(df_fe, target)

        print(f"Training XGBoost for {target}...")
        model, preds, rmse, mape = train_xgboost(X_train, y_train, X_test, y_test)

        print(f"{target} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

        # Save model for later
        model_path = os.path.join(MODEL_DIR, f"xgb_{target}.joblib")
        joblib.dump(model, model_path)

        # Save predictions to DataFrame for output
        pred_df = X_test.copy()
        pred_df["Date"] = df_fe.loc[X_test.index, "Date"].values
        pred_df["actual"] = y_test.values
        pred_df["predicted"] = preds

        # Save prediction CSV
        pred_df.to_csv(os.path.join(MODEL_DIR, f"xgb_preds_{target}.csv"), index=False)

        results.append({
            "target": target,
            "rmse": rmse,
            "mape": mape
        })

        models[target] = model

    print("\n✅ All targets processed.")
    print(pd.DataFrame(results))
