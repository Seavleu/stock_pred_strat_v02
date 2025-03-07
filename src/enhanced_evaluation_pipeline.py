"""
Enhanced Evaluation Pipeline for Stock Price Prediction

Objective:
  - Apply dynamic feature scaling:
      * Normalize returns-based features (e.g., closing_price, Adj Close) using MinMax scaling to [-1, 1]
      * Standardize technical indicators (e.g., highest_price, MA_5, etc.) using Z-score normalization
  - Evaluate predictive signal strength using IC (Pearson correlation) and RIC (Spearman correlation)
  - Integrate these new evaluation metrics into our training pipeline

Note: This script assumes that the dataset contains a single stock’s data. 
For multi-stock data, per-stock normalization should be applied.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
import xgboost as xgb

class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="closing_price"):
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
# 1. Dynamic Feature Scaling Functions
#############################################

def scale_features(df, returns_features, technical_features):
    """
    Scale returns-based features with MinMax scaling (-1, 1) and
    technical indicators with Z-score normalization.
    
    Args:
        df (DataFrame): Input dataset.
        returns_features (list): Columns to scale with MinMax.
        technical_features (list): Columns to standardize using Z-score.
        
    Returns:
        DataFrame: Scaled dataset.
    """
    df_scaled = df.copy()
    # scale returns-based features
    if returns_features:
        scaler_mm = MinMaxScaler(feature_range=(-1, 1))
        df_scaled[returns_features] = scaler_mm.fit_transform(df_scaled[returns_features])
    # standardize technical features
    if technical_features:
        scaler_std = StandardScaler()
        df_scaled[technical_features] = scaler_std.fit_transform(df_scaled[technical_features])
    return df_scaled

#############################################
# 2. IC & RIC Calculation
#############################################

def calculate_ic_ric(y_true, y_pred):
    """
    Calculate the Information Coefficient (IC) and Rank Information Coefficient (RIC)
    using Pearson and Spearman correlation respectively.
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
    Returns:
        ic (float): Pearson correlation coefficient.
        ric (float): Spearman rank correlation coefficient.
    """
    ic, _ = pearsonr(y_true, y_pred)
    ric, _ = spearmanr(y_true, y_pred)
    return ic, ric

#############################################
# 3. Training, Hyperparameter Optimization, and Evaluation (reuse existing functions)
#############################################

def objective(trial, train_loader, input_size):
    seq_length = trial.suggest_int("seq_length", 30, 90)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = 10  # for tuning

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
    # remove any NaNs before calc metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    if np.isnan(predictions).any() or np.isnan(actuals).any():
        raise ValueError("Evaluation data contains NaN values.")
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f"Evaluation Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    # calc IC & RIC
    ic, ric = calculate_ic_ric(actuals, predictions)
    print(f"Information Coefficient (IC): {ic:.4f}, Rank IC (RIC): {ric:.4f}")
    return mse, rmse, r2, mape, ic, ric

#############################################
# 4. Main Pipeline
#############################################

def main(): 
    enhanced_csv = "data/processed/alpha158_enhanced_features.csv"
    if not os.path.exists(enhanced_csv) or os.path.getsize(enhanced_csv) == 0:
        sys.exit(f"Error: {enhanced_csv} is missing or empty. Run the Alpha158-enhanced feature engineering pipeline first.")
    df = pd.read_csv(enhanced_csv, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {enhanced_csv}.")

    # Define feature columns for scaling
    # Assume returns-based features are: 'closing_price' and 'Adj Close'
    # and the technical indicators are all others (excluding 'timestamp')
    all_cols = df.columns.tolist()
    non_feature_cols = ["timestamp", "closing_price"]
    feature_cols = [col for col in all_cols if col not in non_feature_cols]
    
    returns_features = ["Adj Close"] if "Adj Close" in feature_cols else []
    technical_features = [col for col in feature_cols if col not in returns_features]
    
    # dynamic feature scaling
    df_scaled = scale_features(df, returns_features, technical_features)
    scaled_csv = "data/processed/alpha158_enhanced_features_scaled.csv"
    df_scaled.to_csv(scaled_csv, index=False)
    print(f"Scaled dataset saved to {scaled_csv}")

    # prepare
    feature_columns = [col for col in df_scaled.columns if col not in ["timestamp", "closing_price"]]
    seq_length = 30  
    dataset = StockDataset(df_scaled, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = len(feature_columns)
    
    # Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params
    
    # train + best hp
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
    
    # evaluate the model with IC & RIC calcs
    evaluate_model(final_model, train_loader)
    
    print("Enhanced evaluation pipeline complete. The model now uses dynamic feature scaling and is evaluated with IC & RIC metrics.")

if __name__ == "__main__":
    main()
