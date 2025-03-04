'''
This script will perform feature selections, optimize hyperparameters, compare model architecture,
integrate macroeconomics and alternative data (placholder), and evaluates performance.

1. We will try to select the most predictive features using mutual information and SHAP values
2. Then tuning hyperparameters such as seq_length, hidden_size, dropout with Optuna
3. We will also write a function to compare an LSTM model with a baseline XGBoost model
4. As for (Placeholder) integrating macroeconomic & alternative data
''' 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import shap
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from feature_engineering import feature_engineering_pipeline
matplotlib.use("Agg")
#############################################
# 1. Feature Selection Functions
#############################################

def compute_mutual_info(X, y):
    """Compute mutual information scores for each feature."""
    mi_scores = mutual_info_regression(X, y)
    return pd.Series(mi_scores, index=X.columns)

def plot_correlation_matrix(df):
    """Plot a heatmap of the correlation matrix."""
    corr_matrix = df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def perform_feature_selection(engineered_df):
    """
    Select the most predictive features using mutual information and SHAP values.
    Prints MI scores, plots SHAP summary, and returns a DataFrame with selected features.
    """
    # Assume target is 'closing_price' and drop timestamp
    X = engineered_df.drop(columns=["timestamp", "closing_price"])
    y = engineered_df["closing_price"]

    # Compute mutual information scores
    mi_scores = compute_mutual_info(X, y)
    print("Mutual Information Scores:\n", mi_scores.sort_values(ascending=False))

    # Plot correlation matrix to spot redundancy
    plot_correlation_matrix(engineered_df.drop(columns=["timestamp"]))

    # --- Modified SHAP Section ---
    # Sample a subset of the data for SHAP analysis
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_sample, y_sample)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    # Save the SHAP summary plot instead of showing it interactively
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("shap_summary_plot.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary_plot.png'.")

    # For demonstration, select the top 50% of features based on MI score.
    num_selected = max(1, int(len(mi_scores) * 0.5))
    selected_features = mi_scores.sort_values(ascending=False).head(num_selected).index.tolist()
    print("Selected Features:", selected_features)

    # Return DataFrame with timestamp, target, and selected features
    selected_df = engineered_df[["timestamp", "closing_price"] + selected_features]
    return selected_df

#############################################
# 2. Dataset & Model Definitions
#############################################

class StockDataset(Dataset):
    def __init__(self, df, seq_length=30, feature_columns=None, target_column="closing_price"):
        """
        Args:
            df (DataFrame): DataFrame with engineered features.
            seq_length (int): Number of time steps in each sequence.
            feature_columns (list): List of feature columns to use.
            target_column (str): Column name for the target.
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
# 3. Hyperparameter Optimization with Optuna
#############################################

def objective(trial, train_loader, input_size):
    # Suggest hyperparameters
    seq_length = trial.suggest_int("seq_length", 30, 90)
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = 10  # Shorter training for tuning

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
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return epoch_loss

#############################################
# 4. Final Model Training Function
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
        print(f"Final Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    return model

#############################################
# 5. Model Evaluation Function
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
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f"Evaluation Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    return mse, rmse, r2, mape

#############################################
# 6. Main Pipeline
#############################################

def main():
    # Step 1: Load engineered features (generated by feature_engineering.py)
    engineered_csv = "data/processed/engineered_features.csv"
    if not os.path.exists(engineered_csv) or os.path.getsize(engineered_csv) == 0:
        sys.exit(f"Error: {engineered_csv} is missing or empty. Run the feature engineering pipeline first.")
    df = pd.read_csv(engineered_csv, parse_dates=['timestamp'])
    
    # Step 2: Feature Selection - remove weak or redundant features
    selected_df = perform_feature_selection(df)
    selected_csv = "data/processed/selected_features.csv"
    selected_df.to_csv(selected_csv, index=False)
    print(f"Selected features saved to {selected_csv}")
    
    # Step 3: Prepare dataset for training
    feature_columns = selected_df.columns.tolist()
    feature_columns.remove("timestamp")
    feature_columns.remove("closing_price")
    seq_length = 30  # Initial value; will be tuned by Optuna
    dataset = StockDataset(selected_df, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = len(feature_columns)
    
    # Step 4: Hyperparameter Optimization using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    print("Best hyperparameters:", study.best_params)
    
    # Step 5: Train Final Model with Best Hyperparameters
    best_params = study.best_params
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
    
    # Step 6: Evaluate Final Model (here, using training set as a demo; ideally use a separate test set)
    evaluate_model(final_model, train_loader)
    
    # Step 7: Baseline Comparison with XGBoost
    X = selected_df.drop(columns=["timestamp", "closing_price"])
    y = selected_df["closing_price"]
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    xgb_preds = xgb_model.predict(X)
    mse = mean_squared_error(y, xgb_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, xgb_preds)
    mape = mean_absolute_percentage_error(y, xgb_preds)
    print(f"XGBoost Baseline -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    
    # Step 8: (Placeholder) Integrate macroeconomic & alternative data here
    print("Model optimization pipeline complete. Consider integrating macroeconomic data and exploring alternative architectures (e.g., Transformer-based models or ensemble methods).")

if __name__ == "__main__":
    main()
