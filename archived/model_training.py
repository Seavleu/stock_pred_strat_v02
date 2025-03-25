'''
- We updated the StockDataset to use future_avg_return a the target for multi-step forecasting. 
Both LSTM and transformer model are available via a separate `PositionalEncoding` class
- Evaluation metrics: RMSE, R-Sqaure, MAPE, IC, and RIC
- Pipeline: load engineered features, define feature cols, create dataset and DataLoader, perform HP 
tuning, train the final model (transformer), and evaluate the model
''' 
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import optuna
import xgboost as xgb
from feature_refinement import add_future_return_targets
from sklearn.preprocessing import StandardScaler
import joblib

# 🔥 FEATURE SCALING: Normalize dataset using StandardScaler
def scale_features(df, feature_columns):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    joblib.dump(scaler, "scaler.pkl")  # Save scaler for later use in inference
    return df

# 📌 StockDataset: Optimized for multi-step forecasting
class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="future_avg_return"):
        self.seq_length = seq_length
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

# 📌 LSTM Model: Moved to GPU by default
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Transformer Model with positional encoding (if desired)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = torch.tensor(pe, dtype=torch.float32)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


# 📌 Transformer Model: Optimized for efficiency
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, dropout=0.1, output_size=1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_linear(x)  # [batch_size, seq_length, d_model]
        x = x.transpose(0, 1)  # [seq_length, batch_size, d_model]
        x = self.transformer_encoder(x)  # Apply transformer encoder
        out = x[-1, :, :]  # Take output from last time step
        out = self.fc(out)
        return out

#############################################
# Hyperparameter Tuning Objective (using Optuna)
#############################################
def objective(trial, train_loader, input_size, model_type="LSTM"):
    seq_length = trial.suggest_int("seq_length", 30, 90)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "LSTM":
        model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)
    else:
        model = TransformerModel(input_size, d_model=hidden_size, nhead=4,
                                  num_layers=num_layers, dropout=dropout).to(device)
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

def get_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# 📌 Early Stopping Implementation
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience  # Stop training if patience exceeded

# 🔥 Train final model with Optimized Pipeline
def train_final_model(train_loader, input_size, best_params, num_epochs=50, model_type="LSTM"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "LSTM":
        model = LSTMModel(input_size, best_params["hidden_size"], best_params["num_layers"], best_params["dropout"]).to(device)
    else:
        model = TransformerModel(input_size, d_model=best_params["hidden_size"], nhead=2, num_layers=1, dropout=best_params["dropout"]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🔥 Gradient Clipping
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

        if early_stopping.step(epoch_loss):
            print("Early stopping triggered. Training stopped.")
            break

    return model

#############################################
# Evaluation Metrics: IC & RIC
#############################################
def calculate_ic_ric(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr
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
    mse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    ic, ric = calculate_ic_ric(actuals, predictions)
    print(f"Evaluation Metrics -> RMSE: {mse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    print(f"IC: {ic:.4f}, Rank IC: {ric:.4f}")
    return mse, r2, mape, ic, ric

#############################################
# Main Pipeline
#############################################
def main():
    input_csv = "data/processed/refined_features.csv"
    if not os.path.exists(input_csv):
        sys.exit("Error: Engineered features CSV is missing. Run feature_engineering.py first.")

    df = pd.read_csv(input_csv, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {input_csv}.")

    # 🔥 Apply Feature Scaling
    feature_columns = [col for col in df.columns if col not in ["date", "future_avg_return"]]
    df = scale_features(df, feature_columns)

    # Prepare Dataset
    seq_length = 30
    dataset = StockDataset(df, seq_length=seq_length, feature_columns=feature_columns, target_column="future_avg_return")
    train_loader = get_dataloader(dataset, batch_size=32)
    input_size = len(feature_columns)

    # Hyperparameter Tuning with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    best_params = study.best_params

    # Train Final Model
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=50, model_type="Transformer")

    print("Training complete. Evaluating model...")
    evaluate_model(final_model, train_loader)

if __name__ == "__main__":
    main()