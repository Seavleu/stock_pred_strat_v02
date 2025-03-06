"""
Alpha158-Enhanced Pipeline for Korean Stock Price Prediction

This script extends our current pipeline by incorporating additional
Alpha158-inspired features and macroeconomic data, while preserving our
dynamic feature selection approach. It then trains an LSTM model on the
enhanced feature set and compares performance to our dynamic feature set.

Enhancements include:
  1. Regression-based features: BETA, RSQR, RESI (using a placeholder market return)
  2. K-bar features: KMID, KLEN, KSFT (capturing candlestick characteristics)
  3. Enhanced lag analysis: Rolling-window lag statistics (mean & std) over multiple windows
  4. VWAP variations: Different methods for computing VWAP (skipped if trading_volume not available)
  5. Integration of placeholder macroeconomic data (FX rate, KOSDAQ trend, news sentiment)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

#############################################
# 1. Alpha158-inspired Feature Computations
#############################################

def compute_returns(series):
    """Compute daily returns."""
    return series.pct_change()

def compute_regression_features(df, window=30):
    """
    Compute regression-based features (BETA, RSQR, RESI) using a rolling window.
    Here we use a placeholder market return computed as the rolling mean return of 'Adj Close'.
    In practice, we will replace with an actual market index.
    """
    df["stock_return"] = compute_returns(df["Adj Close"])
    df["market_return"] = df["stock_return"].rolling(window=window, min_periods=window).mean()
    
    beta_list = []
    rsqr_list = []
    resi_list = []
    
    for i in range(len(df)):
        if i < window:
            beta_list.append(np.nan)
            rsqr_list.append(np.nan)
            resi_list.append(np.nan)
        else:
            y = df["stock_return"].iloc[i-window:i].values.reshape(-1,1)
            X = df["market_return"].iloc[i-window:i].values.reshape(-1,1)
            if np.isnan(X).any():
                beta_list.append(np.nan)
                rsqr_list.append(np.nan)
                resi_list.append(np.nan)
                continue
            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0][0]
            rsqr = reg.score(X, y)
            y_pred = reg.predict(X)
            resi = np.std(y - y_pred)
            beta_list.append(beta)
            rsqr_list.append(rsqr)
            resi_list.append(resi)
    
    df["BETA"] = beta_list
    df["RSQR"] = rsqr_list
    df["RESI"] = resi_list
    df.drop(columns=["stock_return", "market_return"], inplace=True)
    return df

def compute_kbar_features(df):
    """
    Compute K-bar features:
      KMID: Mid-price = (highest_price + lowest_price) / 2.
             If 'lowest_price' is missing, use 'Rolling_Min_5' as a proxy.
      KLEN: Candle length = highest_price - lowest_price (or proxy).
      KSFT: Shift = closing_price - KMID.
    """
    low_col = "lowest_price" if "lowest_price" in df.columns else "Rolling_Min_5"
    df["KMID"] = (df["highest_price"] + df[low_col]) / 2
    df["KLEN"] = df["highest_price"] - df[low_col]
    df["KSFT"] = df["closing_price"] - df["KMID"]
    return df

def compute_enhanced_lag_features(df, column, windows=[3, 5, 10, 20]):
    """
    Compute rolling lag statistics for a given column.
    For each window, compute the mean and standard deviation of past values.
    """
    for w in windows:
        df[f"{column}_lag_mean_{w}"] = df[column].rolling(window=w, min_periods=1).mean().shift(1)
        df[f"{column}_lag_std_{w}"] = df[column].rolling(window=w, min_periods=1).std().shift(1)
    return df

def compute_vwap_variations(df):
    """
    Compute different VWAP (Volume Weighted Average Price) variants.
    If 'trading_volume' is missing, skip VWAP calculations.
    """
    if "trading_volume" not in df.columns:
        print("trading_volume not found. Skipping VWAP variations.")
        return df
    
    # Use 'lowest_price' if available, otherwise use 'Rolling_Min_5'
    low_col = "lowest_price" if "lowest_price" in df.columns else "Rolling_Min_5"
    df["typical_price"] = (df["highest_price"] + df[low_col] + df["closing_price"]) / 3
    df["vwap_typical"] = (df["typical_price"] * df["trading_volume"]).cumsum() / df["trading_volume"].cumsum()
    
    # For OHLC4 and HLC3, if opening_price is missing, skip them
    if "opening_price" in df.columns:
        df["ohlc4_price"] = (df["opening_price"] + df["highest_price"] + df[low_col] + df["closing_price"]) / 4
        df["vwap_ohlc4"] = (df["ohlc4_price"] * df["trading_volume"]).cumsum() / df["trading_volume"].cumsum()
    
    df["hlc3_price"] = (df["highest_price"] + df[low_col] + df["closing_price"]) / 3
    df["vwap_hlc3"] = (df["hlc3_price"] * df["trading_volume"]).cumsum() / df["trading_volume"].cumsum()
    
    df.drop(columns=["typical_price", "ohlc4_price", "hlc3_price"], errors="ignore", inplace=True)
    return df

def integrate_macroeconomic_data(df):
    """
    Integrate placeholder macroeconomic & alternative data.
    In practice, load real data for FX rates, KOSDAQ trends, news sentiment, etc.
    """
    np.random.seed(42)
    df["FX_rate"] = 1.1 + np.random.normal(0, 0.01, len(df))
    df["KOSDAQ_trend"] = np.linspace(1000, 1200, len(df)) + np.random.normal(0, 5, len(df))
    df["news_sentiment"] = np.random.uniform(-1, 1, len(df))
    return df

def compute_alpha158_features(df):
    """
    Compute all additional features inspired by Alpha158.
    """
    df = compute_regression_features(df, window=30)
    df = compute_kbar_features(df)
    df = compute_enhanced_lag_features(df, "closing_price", windows=[3, 5, 10, 20])
    df = compute_vwap_variations(df)  # This will be skipped if trading_volume is absent
    df = integrate_macroeconomic_data(df)
    return df

#############################################
# 2. Dataset & Model Definitions (reuse existing Classes)
#############################################

class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="closing_price"):
        """
        Args:
            df (DataFrame): DataFrame with features.
            seq_length (int): Number of time steps per sequence.
            feature_columns (list): List of feature columns.
            target_column (str): Target column name.
        """
        self.seq_length = seq_length
        self.feature_columns = feature_columns if feature_columns else df.drop(columns=["timestamp", target_column]).columns.tolist()
        self.data = df.sort_values("timestamp").reset_index(drop=True)
        self.features = self.data[self.feature_columns].values
        self.targets = self.data[target_column].values

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#############################################
# 3. Hyperparameter Optimization with Optuna (reuse existing functions)
#############################################

def objective(trial, train_loader, input_size):
    seq_length = trial.suggest_int("seq_length", 30, 90)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = 10  # for tuning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            epoch_loss /= len(train_loader.dataset)
            
            # If loss is NaN, return a high penalty value
            if np.isnan(epoch_loss):
                return float("inf")
            
            trial.report(epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return epoch_loss
    except Exception as e:
        # If any exception occurs, return infinity to prune the trial.
        print("Exception in objective:", e)
        return float("inf")

def train_final_model(train_loader, input_size, best_params, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size,
                      best_params["hidden_size"],
                      best_params["num_layers"],
                      best_params["dropout"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step(epoch_loss)
        print(f"Final Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    return model

def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).squeeze()
            preds = outputs.cpu().numpy()
            if np.isnan(preds).any():
                print("Warning: NaNs found in predictions")
            predictions.extend(preds)
            actuals.extend(y_batch.cpu().numpy())
    if np.isnan(predictions).any() or np.isnan(actuals).any():
        raise ValueError("Evaluation data contains NaN values.")
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f"Evaluation Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    return mse, rmse, r2, mape

#############################################
# 4. Main Pipeline: Compare Alpha158-Enhanced vs. Dynamic Feature Set
#############################################

def main():
    # Load refined dataset from feature refinement (refined_features.csv)
    refined_csv = "data/processed/refined_features.csv"
    if not os.path.exists(refined_csv) or os.path.getsize(refined_csv) == 0:
        sys.exit(f"Error: {refined_csv} is missing or empty. Run the feature refinement pipeline first.")
    df = pd.read_csv(refined_csv, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {refined_csv}.")

    # Generate Alpha158-enhanced features
    df_alpha = df.copy()
    df_alpha = compute_alpha158_features(df_alpha)
    
    # Extra cleaning: replace infinities and drop remaining NaNs
    df_alpha = df_alpha.replace([np.inf, -np.inf], np.nan)
    before_drop = df_alpha.shape[0]
    df_alpha.dropna(inplace=True)
    after_drop = df_alpha.shape[0]
    print(f"After cleaning, dropped {before_drop - after_drop} rows. Final shape: {df_alpha.shape}")
    
    # Save the enhanced dataset
    alpha_csv = "data/processed/alpha158_enhanced_features.csv"
    df_alpha.to_csv(alpha_csv, index=False)
    print(f"Alpha158-enhanced dataset saved to {alpha_csv}")

    # Load dynamic-selected features (if available); otherwise, use refined_features as baseline
    dynamic_csv = "data/processed/dynamic_selected_features.csv"
    if os.path.exists(dynamic_csv) and os.path.getsize(dynamic_csv) > 0:
        df_dynamic = pd.read_csv(dynamic_csv, parse_dates=["timestamp"])
        print(f"Loaded dynamic-selected dataset with features: {df_dynamic.columns.tolist()}")
    else:
        print("Dynamic-selected features file not found. Using refined_features as dynamic baseline.")
        df_dynamic = df.copy()
    
    # Setup training for both approaches: Alpha158-enhanced & dynamic-selected
    for approach, dataset, label in zip(
        ["Alpha158-enhanced", "Dynamic-selected"],
        [df_alpha, df_dynamic],
        ["Alpha158", "Dynamic"]
    ):
        print(f"\nTraining model with {label} feature set:")
        feature_columns = [col for col in dataset.columns if col not in ["timestamp", "closing_price"]]
        seq_length = 30  # initial value for tuning
        train_dataset = StockDataset(dataset, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        input_size = len(feature_columns)
        
        # Hyperparameter Optimization using Optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
        print("Best hps for", label, ":", study.best_params)
        
        best_params = study.best_params
        final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
        print(f"Evaluation for {label} feature set:")
        evaluate_model(final_model, train_loader)
    
    # Baseline comparison with XGBoost on each dataset
    for approach, dataset in zip(["Alpha158-enhanced", "Dynamic-selected"], [df_alpha, df_dynamic]):
        print(f"\nXGBoost Baseline for {approach} feature set:")
        X = dataset.drop(columns=["timestamp", "closing_price"])
        y = dataset["closing_price"]
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X, y)
        xgb_preds = xgb_model.predict(X)
        mse = mean_squared_error(y, xgb_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, xgb_preds)
        mape = mean_absolute_percentage_error(y, xgb_preds)
        print(f"XGBoost -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    
    print("Alpha158-enhanced pipeline complete. Next steps include further validation, integration of real macroeconomic data, and exploring advanced architectures.")

if __name__ == "__main__":
    main()
