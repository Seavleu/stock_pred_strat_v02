"""
Alpha158-Enhanced Pipeline for Korean Stock Prediction with Robust Validation & Backtesting

Objective:
  - Dynamically select and scale features (using per-stock normalization) and market-wide indicators.
  - Integrate market-wide indicators (VWAP variations, KOSPI trends, macroeconomic factors).
  - Implement robust validation (80/10/10 split) and backtesting.
  - Utilize TSDataSampler for efficient mini-batching.
  - Retain our LSTM Seq2Seq + Attention model and Transformer-based models for multi-stock scenarios.
  - Evaluate with IC/RIC metrics.
  - Use swifter to speed up groupby apply operations for per-stock normalization.

Modifications are marked with **"Modification:"**.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import swifter  
import matplotlib
matplotlib.use("Agg")
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
from backtesting import backtest_model  
# comment it out to process dataframes sequentially
import ray
ray.init(
    ignore_reinit_error=True,# this makes sure ray is initialized only once
    num_cpus=10,  # Adjust based on your system
    object_store_memory= 4 * 10**9  # 4GB memory limit
)
import swifter # for parallel processing of dataframes
# comment out until here to not use parallelization
import concurrent.futures # for loading dataframes in parallel

# Fix the import path for log_config
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_config import setup_logger, get_logger # logging handlers.

# logs will be saved in a file called app.log
setup_logger(log_file_path="app.log")
logger = get_logger()
#############################################
# Modification: TSDataSampler for Efficient Mini-Batching
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
# 1. Alpha158-inspired & Market-Wide Feature Computations (unchanged)
#############################################

def compute_returns(series):
    return series.pct_change()

def compute_regression_features(df, window=30):
    df["stock_return"] = compute_returns(df["closing_price"])  # Modification: Use closing_price
    df["market_return"] = df["stock_return"].rolling(window=window, min_periods=window).mean()
    beta_list, rsqr_list, resi_list = [], [], []
    for i in range(len(df)):
        if i < window:
            beta_list.append(np.nan); rsqr_list.append(np.nan); resi_list.append(np.nan)
        else:
            y = df["stock_return"].iloc[i-window:i].values.reshape(-1,1)
            X = df["market_return"].iloc[i-window:i].values.reshape(-1,1)
            if np.isnan(X).any():
                beta_list.append(np.nan); rsqr_list.append(np.nan); resi_list.append(np.nan)
                continue
            reg = LinearRegression().fit(X, y)
            beta_list.append(reg.coef_[0][0])
            rsqr_list.append(reg.score(X, y))
            y_pred = reg.predict(X)
            resi_list.append(np.std(y - y_pred))
    df["BETA"] = beta_list; df["RSQR"] = rsqr_list; df["RESI"] = resi_list
    df.drop(columns=["stock_return", "market_return"], inplace=True)
    return df

def compute_kbar_features(df):
    low_col = "lowest_price" if "lowest_price" in df.columns else "Rolling_Min_5"
    df["KMID"] = (df["highest_price"] + df[low_col]) / 2
    df["KLEN"] = df["highest_price"] - df[low_col]
    df["KSFT"] = df["closing_price"] - df["KMID"]
    return df

def compute_enhanced_lag_features(df, column, windows=[3, 5, 10, 20]):
    for w in windows:
        df[f"{column}_lag_mean_{w}"] = df[column].rolling(window=w, min_periods=1).mean().shift(1)
        df[f"{column}_lag_std_{w}"] = df[column].rolling(window=w, min_periods=1).std().shift(1)
    return df

def compute_vwap_variations(df):
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
    np.random.seed(42)
    df["FX_rate"] = 1.1 + np.random.normal(0, 0.01, len(df))
    df["KOSDAQ_trend"] = np.linspace(1000, 1200, len(df)) + np.random.normal(0, 5, len(df))
    df["news_sentiment"] = np.random.uniform(-1, 1, len(df))
    return df

def compute_market_indicators(df, window_list=[5,10,20,30,60]):
    np.random.seed(42)
    df["KOSPI_close"] = 3000 + np.cumsum(np.random.normal(0, 10, len(df)))
    df["KOSPI_volume"] = 1e6 + np.random.normal(0, 50000, len(df))
    df["KOSPI_return"] = df["KOSPI_close"].pct_change()
    for w in window_list:
        df[f"KOSPI_return_mean_{w}"] = df["KOSPI_return"].rolling(window=w, min_periods=1).mean()
        df[f"KOSPI_volume_mean_{w}"] = df["KOSPI_volume"].rolling(window=w, min_periods=1).mean()
    return df

def compute_alpha158_features(df):
    df = compute_regression_features(df, window=30)
    df = compute_kbar_features(df)
    df = compute_enhanced_lag_features(df, "closing_price", windows=[3,5,10,20])
    df = compute_vwap_variations(df)
    df = integrate_macroeconomic_data(df)
    df = compute_market_indicators(df)
    return df

#############################################
# 2. Dataset & Model Definitions (adj per-stock normalization)
#############################################

class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="closing_price"):
        self.seq_length = seq_length
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
# 3. HP Opt with Optuna  
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
# train-val-test split (80/10/10)
#############################################
def train_val_test_split_ts(df, train_size=0.8, val_size=0.1, test_size=0.1):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

#############################################
# 4. Main Pipeline: Robust Validation, Multi-Stock 
#############################################

def main():
    logger.info("Starting Alpha158 enhanced pipeline")
    # Load raw extracted stock files with company info
    raw_path = "data/raw/korean_stock_extracted/"
    file_pattern = os.path.join(raw_path, "*_stock_data.csv")
    file_list = glob.glob(file_pattern)
    if not file_list:
        logger.error("No extracted stock data files found in data/raw/korean_stock_extracted/")
        sys.exit("Error: No extracted stock data files found in data/raw/korean_stock_extracted/")
        
    logger.info(f"Found {len(file_list)} stock data files")
    
    def load_file(file_path):
        logger.info(f"Loading file: {file_path}")
        return pd.read_csv(file_path, parse_dates=["timestamp"])
    # load files in parallel 
    # use concurrent futures because it's an I/O operation
    df_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        df_list = list(executor.map(load_file, file_list))
    df_raw = pd.concat(df_list, ignore_index=True)
    logger.info(f"Loaded raw data from {len(file_list)} files; combined shape: {df_raw.shape}")
    print(f"Loaded raw data from {len(file_list)} files; combined shape: {df_raw.shape}")
    del df_list # clean up memory

    logger.info("Processing data per company")
    
    # Process data per company using compute_alpha158_features
    df_processed_list = []
    for company, group in df_raw.groupby("company"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        group = compute_alpha158_features(group)
        group.dropna(inplace=True)
        df_processed_list.append(group)
    df_processed = pd.concat(df_processed_list, ignore_index=True)
    print(f"After processing per company, shape: {df_processed.shape}")
    
    # Per-stock normalization: apply scaling per company
    def scale_group(group):
        all_cols = group.columns.tolist()
        non_feature_cols = ["timestamp", "closing_price", "company"]
        feature_cols = [col for col in all_cols if col not in non_feature_cols]
        returns_features = ["closing_price"]
        technical_features = [col for col in feature_cols if col not in returns_features]
        if returns_features:
            scaler_mm = MinMaxScaler(feature_range=(-1, 1))
            group[returns_features] = scaler_mm.fit_transform(group[returns_features])
        if technical_features:
            scaler_std = StandardScaler()
            group[technical_features] = scaler_std.fit_transform(group[technical_features])
        return group
    df_scaled = df_processed.swifter.groupby("company", group_keys=False).apply(scale_group)
    scaled_csv = "data/processed/alpha158_enhanced_features_scaled.csv"
    df_scaled.to_csv(scaled_csv, index=False)
    print(f"Scaled dataset saved to {scaled_csv}")

    # Progressive Training: choose one company to train first
    company_to_train = df_scaled["company"].unique()[0]
    df_company = df_scaled[df_scaled["company"] == company_to_train].copy()
    print(f"Training on company: {company_to_train}, shape: {df_company.shape}")
    
    # Modification: Split into Train (80%), Validation (10%), and Test (10%)
    train_df, val_df, test_df = train_val_test_split_ts(df_company, train_size=0.8, val_size=0.1, test_size=0.1)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    # Prepare training, validation, and test datasets
    feature_columns = [col for col in df_company.columns if col not in ["timestamp", "closing_price", "company"]]
    seq_length = 30
    train_dataset = StockDataset(train_df, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    val_dataset = StockDataset(val_df, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    test_dataset = StockDataset(test_df, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    batch_size = 64
    train_sampler = TSDataSampler(train_dataset, batch_size=batch_size)
    val_sampler = TSDataSampler(val_dataset, batch_size=batch_size)
    test_sampler = TSDataSampler(test_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    input_size = len(feature_columns)

    # Hyperparameter optimization using Optuna on the training set
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params

    # Train final model on the training set
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
    
    # Evaluate on training, validation, and test sets
    print("\nEvaluation on Training Set:")
    evaluate_model(final_model, train_loader)
    print("\nEvaluation on Validation Set:")
    evaluate_model(final_model, val_loader)
    print("\nEvaluation on Test Set:")
    evaluate_model(final_model, test_loader)

    # Modification: Use the separate backtesting module to backtest on the Test set.
    print("\nBacktesting on Test Set:")
    backtest_model(final_model, test_loader, initial_capital=1000000, threshold=0.0)

    print("Enhanced evaluation pipeline complete. Next steps: expand multi-stock training, integrate real macroeconomic data, and explore advanced architectures.")

if __name__ == "__main__":
    main()
