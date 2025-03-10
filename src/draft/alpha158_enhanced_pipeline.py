"""
Alpha158-Enhanced Pipeline for Korean Stock Prediction with Market Awareness and Per-Stock Normalization

Objective:
  - Dynamically select and scale features (using per-stock normalization).
  - Integrate market-wide indicators (VWAP variations, KOSPI trends, macroeconomic factors).
  - Support a progressive training strategy (starting with one stock, later expanding).
  - Retain our LSTM Seq2Seq + Attention model (or consider Transformer-based models for multi-stock scenarios).
  - Evaluate using enhanced metrics including IC (Pearson) and RIC (Spearman).

Modifications are marked with **"Modification:"** in comments.
"""

import os
import sys
import glob  # Modification: For loading multiple raw files
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#############################################
# Modification: TSDataSampler for efficient mini-batching
#############################################
class TSDataSampler(Sampler):
    """Yields sequential batches for time-series data."""
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i+self.batch_size]

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

#############################################
# 1. Alpha158-inspired & Market-Wide Feature Computations
#############################################

def compute_returns(series):
    """Compute daily returns."""
    return series.pct_change()

def compute_regression_features(df, window=30):
    """
    Compute regression-based features (BETA, RSQR, RESI) using a rolling window.
    Modification: Use 'closing_price' (raw data) since 'Adj Close' is not available.
    """
    df["stock_return"] = compute_returns(df["closing_price"])
    df["market_return"] = df["stock_return"].rolling(window=window, min_periods=window).mean()
    
    beta_list, rsqr_list, resi_list = [], [], []
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
            beta_list.append(reg.coef_[0][0])
            rsqr_list.append(reg.score(X, y))
            y_pred = reg.predict(X)
            resi_list.append(np.std(y - y_pred))
    df["BETA"] = beta_list
    df["RSQR"] = rsqr_list
    df["RESI"] = resi_list
    df.drop(columns=["stock_return", "market_return"], inplace=True)
    return df

def compute_kbar_features(df):
    """
    Compute K-bar features:
      KMID: (highest_price + lowest_price) / 2.
      Modification: If 'lowest_price' is missing, use 'Rolling_Min_5' as proxy.
      KLEN: highest_price - lowest_price (or proxy).
      KSFT: closing_price - KMID.
    """
    low_col = "lowest_price" if "lowest_price" in df.columns else "Rolling_Min_5"
    df["KMID"] = (df["highest_price"] + df[low_col]) / 2
    df["KLEN"] = df["highest_price"] - df[low_col]
    df["KSFT"] = df["closing_price"] - df["KMID"]
    return df

def compute_enhanced_lag_features(df, column, windows=[3, 5, 10, 20]):
    """Compute rolling lag statistics for a column (mean and std)."""
    for w in windows:
        df[f"{column}_lag_mean_{w}"] = df[column].rolling(window=w, min_periods=1).mean().shift(1)
        df[f"{column}_lag_std_{w}"] = df[column].rolling(window=w, min_periods=1).std().shift(1)
    return df

def compute_vwap_variations(df):
    """
    Compute VWAP variations.
    Modification: Skip VWAP if 'trading_volume' is not available.
    """
    if "trading_volume" not in df.columns:
        print("trading_volume not found. Skipping VWAP variations.")
        return df
    low_col = "lowest_price" if "lowest_price" in df.columns else "Rolling_Min_5"
    df["typical_price"] = (df["highest_price"] + df[low_col] + df["closing_price"]) / 3
    df["vwap_typical"] = (df["typical_price"] * df["trading_volume"]).cumsum() / df["trading_volume"].cumsum()
    if "opening_price" in df.columns:
        df["ohlc4_price"] = (df["opening_price"] + df["highest_price"] + df[low_col] + df["closing_price"]) / 4
        df["vwap_ohlc4"] = (df["ohlc4_price"] * df["trading_volume"]).cumsum() / df["trading_volume"].cumsum()
    df["hlc3_price"] = (df["highest_price"] + df[low_col] + df["closing_price"]) / 3
    df["vwap_hlc3"] = (df["hlc3_price"] * df["trading_volume"]).cumsum() / df["trading_volume"].cumsum()
    df.drop(columns=["typical_price", "ohlc4_price", "hlc3_price"], errors="ignore", inplace=True)
    return df

def integrate_macroeconomic_data(df):
    """Integrate placeholder macroeconomic data."""
    np.random.seed(42)
    df["FX_rate"] = 1.1 + np.random.normal(0, 0.01, len(df))
    df["KOSDAQ_trend"] = np.linspace(1000, 1200, len(df)) + np.random.normal(0, 5, len(df))
    df["news_sentiment"] = np.random.uniform(-1, 1, len(df))
    return df

def compute_market_indicators(df, window_list=[5,10,20,30,60]):
    """
    Compute market-wide indicators using simulated KOSPI data.
    Modification: Simulate KOSPI index, compute rolling means for returns and volume.
    """
    np.random.seed(42)
    df["KOSPI_close"] = 3000 + np.cumsum(np.random.normal(0, 10, len(df)))
    df["KOSPI_volume"] = 1e6 + np.random.normal(0, 50000, len(df))
    df["KOSPI_return"] = df["KOSPI_close"].pct_change()
    for w in window_list:
        df[f"KOSPI_return_mean_{w}"] = df["KOSPI_return"].rolling(window=w, min_periods=1).mean()
        df[f"KOSPI_volume_mean_{w}"] = df["KOSPI_volume"].rolling(window=w, min_periods=1).mean()
    return df

def compute_alpha158_features(df):
    """
    Compute Alpha158-inspired features and integrate market-wide indicators.
    """
    df = compute_regression_features(df, window=30)
    df = compute_kbar_features(df)
    df = compute_enhanced_lag_features(df, "closing_price", windows=[3, 5, 10, 20])
    df = compute_vwap_variations(df)  # Will be skipped if trading_volume is missing.
    df = integrate_macroeconomic_data(df)
    df = compute_market_indicators(df)
    return df

#############################################
# 2. Dataset & Model Definitions (existing classes, modified for per-stock normalization)
#############################################

class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="closing_price"):
        """
        Args:
            df (DataFrame): Processed DataFrame.
            seq_length (int): Number of time steps per sequence.
            feature_columns (list): List of feature columns.
            target_column (str): Target column name.
        """
        self.seq_length = seq_length
        # Modification: Remove 'company' from features if present.
        non_feature_cols = ["timestamp", "closing_price", "company"]
        if "company" in df.columns:
            self.feature_columns = feature_columns if feature_columns else df.drop(columns=non_feature_cols).columns.tolist()
        else:
            self.feature_columns = feature_columns if feature_columns else df.drop(columns=["timestamp", "closing_price"]).columns.tolist()
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
# 3. Hyperparameter Optimization with Optuna (existing functions)
#############################################

def objective(trial, train_loader, input_size):
    seq_length = trial.suggest_int("seq_length", 30, 90)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
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
        if np.isnan(epoch_loss):
            return float("inf")
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return epoch_loss

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

def calculate_ic_ric(y_true, y_pred):
    ic, _ = pearsonr(y_true, y_pred)
    ric, _ = spearmanr(y_true, y_pred)
    return ic, ric

def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    if np.isnan(predictions).any() or np.isnan(actuals).any():
        raise ValueError("Evaluation data contains NaN values.")
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    ic, ric = calculate_ic_ric(actuals, predictions)
    print(f"Evaluation Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    print(f"Information Coefficient (IC): {ic:.4f}, Rank IC (RIC): {ric:.4f}")
    return mse, rmse, r2, mape, ic, ric

#############################################
# 4. Main Pipeline: Integration of Dynamic Scaling, Market Indicators, and Progressive Training
#############################################

def main():
    # Modification: Load raw extracted stock files with company info
    raw_path = "data/raw/korean_stock_extracted/"
    file_pattern = os.path.join(raw_path, "*_stock_data.csv")
    file_list = glob.glob(file_pattern)
    if not file_list:
        sys.exit("Error: No extracted stock data files found in data/raw/korean_stock_extracted/")
    df_list = []
    for f in file_list:
        temp_df = pd.read_csv(f, parse_dates=["timestamp"])
        df_list.append(temp_df)
    df_raw = pd.concat(df_list, ignore_index=True)
    print(f"Loaded raw data from {len(file_list)} files; combined shape: {df_raw.shape}")

    # Modification: Process data per company using compute_alpha158_features
    df_processed_list = []
    for company, group in df_raw.groupby("company"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group = compute_alpha158_features(group)
        group.dropna(inplace=True)
        df_processed_list.append(group)
    df_processed = pd.concat(df_processed_list, ignore_index=True)
    print(f"After processing per company, shape: {df_processed.shape}")
    
    # Modification: Per-stock normalization
    def scale_group(group):
        all_cols = group.columns.tolist()
        non_feature_cols = ["timestamp", "closing_price", "company"]
        feature_cols = [col for col in all_cols if col not in non_feature_cols]
        # For returns-based features, we use MinMax scaling (-1,1)
        returns_features = ["closing_price"]  # or add other price-based features if available
        technical_features = [col for col in feature_cols if col not in returns_features]
        if returns_features:
            scaler_mm = MinMaxScaler(feature_range=(-1, 1))
            group[returns_features] = scaler_mm.fit_transform(group[returns_features])
        if technical_features:
            scaler_std = StandardScaler()
            group[technical_features] = scaler_std.fit_transform(group[technical_features])
        return group
    df_scaled = df_processed.groupby("company", group_keys=False).apply(scale_group)
    scaled_csv = "data/processed/alpha158_enhanced_features_scaled.csv"
    df_scaled.to_csv(scaled_csv, index=False)
    print(f"Scaled dataset saved to {scaled_csv}")

    # Modification: Integrate market-wide indicators are already computed in compute_market_indicators
    
    # Modification: Progressive Training Strategy - Train on a single company first
    company_to_train = df_scaled["company"].unique()[0]
    df_train = df_scaled[df_scaled["company"] == company_to_train].copy()
    print(f"Training on company: {company_to_train}, shape: {df_train.shape}")

    feature_columns = [col for col in df_train.columns if col not in ["timestamp", "closing_price", "company"]]
    seq_length = 30  # initial value
    dataset = StockDataset(df_train, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    batch_size = 64
    sampler = TSDataSampler(dataset, batch_size=batch_size)
    train_loader = DataLoader(dataset, batch_sampler=sampler)
    input_size = len(feature_columns)

    # Hyperparameter optimization using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params

    # Train final model using best hyperparameters
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)

    # Evaluate model using IC & RIC alongside traditional metrics
    evaluate_model(final_model, train_loader)

    print("Enhanced evaluation pipeline complete. Next steps: validate on hold-out data, integrate real macroeconomic data, and explore Transformer-based models for multi-stock training.")

if __name__ == "__main__":
    main()
