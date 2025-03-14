"""
multi-stock evaluation pipeline for stock price prediction

objective:
  - load the engineered features for multiple stocks,
  - apply per-stock normalization,
  - split the data (80/10/10) preserving time order,
  - train an lstm model (using future_avg_return as target),
  - evaluate predictive performance with ic & ric,
  - and run backtesting on the test set.

note: this pipeline is intended for multi-stock data.
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
import xgboost as xgb
from backtesting import backtest_model

#############################################
# stock dataset (multi-stock) using 'date' and target 'future_avg_return'
#############################################
class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="future_avg_return"):
        self.seq_length = seq_length
        non_feature_cols = ["date", target_column, "company"]
        if "company" in df.columns:
            self.feature_columns = feature_columns if feature_columns else df.drop(columns=non_feature_cols).columns.tolist()
        else:
            self.feature_columns = feature_columns if feature_columns else df.drop(columns=["date", target_column]).columns.tolist()
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
# multi-stock normalization function
#############################################
def process_multi_stock_data(df):
    """
    apply per-stock normalization to a multi-stock dataset.
    for each company, normalize returns-based features (close, future_avg_return)
    using minmax scaling, and standardize technical features using z-score
    
    returns:
        combined dataframe with per-stock normalization applied.
    """
    def normalize_group(group):
        group = group.sort_values("date").reset_index(drop=True) 
        scaler_mm = MinMaxScaler(feature_range=(-1, 1))
        group[['close', 'future_avg_return']] = scaler_mm.fit_transform(group[['close', 'future_avg_return']])
        # normalize other numeric features (excluding date, company, close, future_avg_return)
        numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
        technical_cols = [col for col in numeric_cols if col not in ['close', 'future_avg_return']]
        if technical_cols:
            scaler_std = StandardScaler()
            group[technical_cols] = scaler_std.fit_transform(group[technical_cols])
        return group

    df_normalized = df.swifter.groupby("company", group_keys=False).apply(normalize_group).reset_index(drop=True)
    return df_normalized

#############################################
# train-validation-test split 80 10 10
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
# ic & ric calc 
#############################################
def calculate_ic_ric(y_true, y_pred):
    ic, _ = pearsonr(y_true, y_pred)
    ric, _ = spearmanr(y_true, y_pred)
    return ic, ric

#############################################
# optuna objective function
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

#############################################
# training final model
#############################################
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

#############################################
# evaluation
#############################################
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
# main pipeline
#############################################
def main(): 
    engineered_csv = "data/processed/refined_features.csv"
    if not os.path.exists(engineered_csv) or os.path.getsize(engineered_csv) == 0:
        sys.exit(f"error: {engineered_csv} is missing or empty. please run the feature engineering pipeline first.")
    df = pd.read_csv(engineered_csv, parse_dates=["date"])
    print(f"loaded {len(df)} rows from {engineered_csv}.")

    # apply per-stock normalization on multi-stock data
    df_multi = process_multi_stock_data(df)
    processed_csv = "data/processed/refined_features_multistock.csv"
    df_multi.to_csv(processed_csv, index=False)
    print(f"multi-stock normalized dataset saved to {processed_csv}")
    
    train_df, val_df, test_df = train_val_test_split_ts(df_multi, train_size=0.8, val_size=0.1, test_size=0.1)
    print(f"multi-stock train shape: {train_df.shape}, val shape: {val_df.shape}, test shape: {test_df.shape}")
    
    # define feature columns (exclude date, future_avg_return, company)
    feature_columns = [col for col in df_multi.columns if col not in ["date", "future_avg_return", "company"]]
    seq_length = 30
    train_dataset = StockDataset(train_df, seq_length=seq_length, feature_columns=feature_columns, target_column="future_avg_return")
    val_dataset = StockDataset(val_df, seq_length=seq_length, feature_columns=feature_columns, target_column="future_avg_return")
    test_dataset = StockDataset(test_df, seq_length=seq_length, feature_columns=feature_columns, target_column="future_avg_return")
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    input_size = len(feature_columns)
    
    # hyperparameter tuning using optuna on training set
    print("optimizing hyperparameters using optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    print("best hyperparameters:", study.best_params)
    best_params = study.best_params
    
    # train final model on training set
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
    
    # evaluate on train, validation, and test sets
    print("\nevaluation on training set:")
    evaluate_model(final_model, train_loader)
    print("\nevaluation on validation set:")
    evaluate_model(final_model, val_loader)
    print("\nevaluation on test set:")
    evaluate_model(final_model, test_loader)
    
    # backtesting on test set using the final model
    print("\nbacktesting on test set:")
    backtest_model(final_model, test_loader, initial_capital=1000000, threshold=0.0)
    
    print("unified multi-stock training pipeline complete. next steps: integrate real macroeconomic data, monitor model performance, and explore further advanced architectures.")

if __name__ == "__main__":
    main()
