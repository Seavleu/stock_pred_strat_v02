"""
enhanced evaluation pipeline for stock price prediction

objective:
  - apply dynamic feature scaling:
      * normalize returns-based features (e.g., close, adj close) using minmax scaling to [-1, 1]
      * standardize technical indicators (e.g., highest_price, MA_5, etc.) using z-score normalization
  - evaluate predictive signal strength using ic (pearson correlation) and ric (spearman correlation)
  - integrate these new evaluation metrics into our training pipeline

note: this script assumes that the dataset contains a single stock’s data.
for multi-stock data, per-stock normalization should be applied.
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

#############################################
# stock dataset for single-stock data (using "date" & target "future_avg_return")
#############################################
class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="future_avg_return"):
        self.seq_length = seq_length
        # exclude date and target columns from features
        non_feature_cols = ["date", target_column]
        self.feature_columns = feature_columns if feature_columns else df.drop(columns=non_feature_cols).columns.tolist()
        self.data = df.sort_values("date").reset_index(drop=True)
        self.features = self.data[self.feature_columns].values
        self.targets = self.data[target_column].values

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#############################################
# lstm model
#############################################
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
# 1. dynamic feature scaling
#############################################
def scale_features(df, returns_features, technical_features):
    """
    using minmax scaling, and standardize technical features using z-score
    """
    df_scaled = df.copy()
    if returns_features:
        scaler_mm = MinMaxScaler(feature_range=(-1, 1))
        df_scaled[returns_features] = scaler_mm.fit_transform(df_scaled[returns_features])
    if technical_features:
        scaler_std = StandardScaler()
        df_scaled[technical_features] = scaler_std.fit_transform(df_scaled[technical_features])
    return df_scaled

#############################################
# 2. ic & ric calc 
#############################################
def calculate_ic_ric(y_true, y_pred):
    ic, _ = pearsonr(y_true, y_pred)
    ric, _ = spearmanr(y_true, y_pred)
    return ic, ric

#############################################
# 3. training, hyperparameter optimization, and evaluation
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
        print(f"final training - epoch [{epoch+1}/{num_epochs}], loss: {epoch_loss:.6f}")
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
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    if np.isnan(predictions).any() or np.isnan(actuals).any():
        raise ValueError("evaluation data contains nan values.")
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f"evaluation metrics -> mse: {mse:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}, mape: {mape:.4f}")
    ic, ric = calculate_ic_ric(actuals, predictions)
    print(f"information coefficient (ic): {ic:.4f}, rank ic (ric): {ric:.4f}")
    return mse, rmse, r2, mape, ic, ric

#############################################
# 4. train-val-test split (80/10/10)
#############################################
def train_val_test_split_ts(df, train_size=0.8, val_size=0.1, test_size=0.1):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

#############################################
# 5. Main Pipeline
#############################################
def main():
    enhanced_csv = "data/interim/refined_features.csv"
    if not os.path.exists(enhanced_csv) or os.path.getsize(enhanced_csv) == 0:
        sys.exit(f"error: {enhanced_csv} is missing or empty. please run the feature engineering -> feature_refinement pipeline first.")
    df = pd.read_csv(enhanced_csv, parse_dates=["date"])
    print(f"loaded {len(df)} rows from {enhanced_csv}.")

    # if multiple stocks are present, filter to a single stock (for single-stock evaluation)
    if "company" in df.columns:
        companies = df["company"].unique()
        print("found companies:", companies)
        selected_company = companies[0]
        print("using data for company:", selected_company)
        df = df[df["company"] == selected_company].copy()

    all_cols = df.columns.tolist()
    non_feature_cols = ["date", "future_avg_return"]
    feature_cols = [col for col in all_cols if col not in non_feature_cols]
    
    returns_features = []   
    technical_features = feature_cols   
    
    # dynamic feature scaling
    df_scaled = scale_features(df, returns_features, technical_features)
    scaled_csv = "data/interim/refined_features_scaled.csv"
    df_scaled.to_csv(scaled_csv, index=False)
    print(f"scaled dataset saved to {scaled_csv}")

    # prepare dataset for training; target is future_avg_return
    feature_columns = [col for col in df_scaled.columns if col not in ["date", "future_avg_return"]]
    seq_length = 30  
    dataset = StockDataset(df_scaled, seq_length=seq_length, feature_columns=feature_columns, target_column="future_avg_return")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = len(feature_columns)
    
    # optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    print("best hyperparameters:", study.best_params)
    best_params = study.best_params
    
    # train on training set
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
    
    evaluate_model(final_model, train_loader)
    
    print("enhanced evaluation pipeline complete. the model now uses dynamic feature scaling and is evaluated with ic & ric metrics.")

if __name__ == "__main__":
    main()
