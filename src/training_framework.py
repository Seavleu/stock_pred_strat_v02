"""
Improved Training Framework for Korean Stock Price Prediction

This script integrates dynamic feature selection, progressive training, and an enhanced
transformer-based architecture to improve stock price forecasting for the Korean stock market.
Data is loaded from individual company CSV files in 'data/interim/korean_stock_extracted/'.
Adaptive scaling is performed at the company level, and the top (most volatile) company is used
for progressive training. A placeholder for rolling-window SHAP-based feature selection is included.
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Data Loading & Preprocessing
# -------------------------

def load_company_data(data_dir="data/raw/korean_stock_extracted"):
    """
    Load CSV files from the specified directory.
    Compute volatility for each company (using standard deviation of daily returns).
    Return a dictionary mapping company name to its DataFrame and select the company with highest volatility.
    """
    files = glob.glob(os.path.join(data_dir, "*_stock_data.csv"))
    if not files:
        sys.exit(f"No stock data files found in {data_dir}.")
    
    company_data = {}
    volatility_scores = {}
    
    for file in files:
        df = pd.read_csv(file, parse_dates=["timestamp"])
        # Ensure data is sorted
        df.sort_values("timestamp", inplace=True)
        # Compute daily returns and volatility as std deviation of returns
        df["return"] = df["closing_price"].pct_change()
        vol = df["return"].std()
        # Extract company name from file name
        company_name = os.path.basename(file).replace("_stock_data.csv", "")
        company_data[company_name] = df
        volatility_scores[company_name] = vol
        print(f"Loaded {company_name}: volatility = {vol:.4f}")
    
    # Select the company with the highest volatility
    top_company = max(volatility_scores, key=volatility_scores.get)
    print(f"Selected top company for training: {top_company} (volatility = {volatility_scores[top_company]:.4f})")
    return top_company, company_data[top_company]

def adaptive_scaling(df):
    """
    Apply RobustScaler to numeric columns on a per-company basis.
    Returns a DataFrame with scaled numeric features.
    """
    scaler = RobustScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Optionally, exclude 'return' if not needed
    if "return" in numeric_cols:
        numeric_cols.remove("return")
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# -------------------------
# Dynamic Feature Selection (Placeholder)
# -------------------------

def dynamic_feature_selection(df, window_size=30):
    """
    Placeholder for dynamic feature selection using rolling-window SHAP-based feature importance.
    In a production system, you would compute SHAP values over rolling windows and select features dynamically.
    For now, we simply return a predefined set of features.
    
    Expected features might include:
      - 'Adj Close', 'highest_price', 'Rolling_Max_5', 'Rolling_Min_5',
        'MA_5', 'MA_50', 'BB_Lower', 'closing_price_lag_3'
    """
    # Here we assume these 8 features were refined previously.
    selected_features = ['Adj Close', 'highest_price', 'Rolling_Max_5', 'Rolling_Min_5',
                         'MA_5', 'MA_50', 'BB_Lower', 'closing_price_lag_3']
    # Ensure the features exist in df; if not, use available ones.
    available_features = [f for f in selected_features if f in df.columns]
    print("Dynamically selected features:", available_features)
    return df[["timestamp", "closing_price"] + available_features]

# -------------------------
# Custom Dataset for Time-Series Data
# -------------------------

class StockDataset(Dataset):
    def __init__(self, df, seq_length=90, feature_columns=None, target_column="closing_price"):
        """
        Args:
            df (DataFrame): DataFrame with selected features.
            seq_length (int): Number of time steps in each sequence.
            feature_columns (list): List of feature columns to use.
            target_column (str): Column name for the target.
        """
        self.seq_length = seq_length
        # If not provided, all columns except timestamp and target are features.
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

# -------------------------
# Enhanced Transformer Architecture
# -------------------------

class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1, output_size=1):
        """
        Transformer-based model for time-series forecasting.
        
        Args:
            input_size (int): Number of input features per time step.
            d_model (int): Dimension of the model (embedding dimension).
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
            output_size (int): Dimension of the model output.
        """
        super(StockTransformer, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x: [batch, seq_length, input_size]
        x = self.input_linear(x)  # [batch, seq_length, d_model]
        # Transformer expects shape [seq_length, batch, d_model]
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)  # [seq_length, batch, d_model]
        # Use the output corresponding to the final time step
        final_out = transformer_out[-1, :, :]  # [batch, d_model]
        output = self.fc_out(final_out)  # [batch, output_size]
        return output

# -------------------------
# Training Functions
# -------------------------

def train_model(model, train_loader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    return model

def evaluate_model(model, data_loader):
    model.eval()
    predictions, actuals = [], []
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

# -------------------------
# Main Training Pipeline
# -------------------------

def main():
    # Step 1: Load individual company data
    company_name, df = load_company_data()
    print(f"Training will proceed on data for {company_name}.")

    # Step 2: Adaptive scaling (scale each company's data individually)
    df = adaptive_scaling(df)
    
    # Step 3: Dynamic feature selection (using placeholder dynamic_feature_selection)
    df_selected = dynamic_feature_selection(df)
    
    # Optionally, you can also run the previously defined feature_engineering_pipeline here if needed.
    # df_engineered = feature_engineering_pipeline(df)  # if you want to enrich the data further
    
    # Step 4: Create time-series dataset (progressive training on one stock)
    # Use a longer sequence length (e.g., 90 days) as suggested by hyperparameter tuning
    seq_length = 90
    # Use the selected features (excluding timestamp and target)
    feature_columns = df_selected.columns.tolist()
    feature_columns.remove("timestamp")
    feature_columns.remove("closing_price")
    dataset = StockDataset(df_selected, seq_length=seq_length, feature_columns=feature_columns, target_column="closing_price")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = len(feature_columns)
    
    # Step 5: Initialize and train the Transformer-based model
    print("Training Transformer-based model...")
    transformer_model = StockTransformer(input_size=input_size, d_model=64, nhead=4, num_layers=4,
                                          dim_feedforward=128, dropout=0.1, output_size=1)
    transformer_model = train_model(transformer_model, train_loader, num_epochs=20, learning_rate=0.005)
    
    # Step 6: Evaluate the trained model
    evaluate_model(transformer_model, train_loader)
    
    # Step 7: Progressive Training Strategy (Placeholder)
    # Here you can later incorporate batch-wise training across multiple companies.
    # For now, training on the top volatile company serves as a starting point.
    
    # Step 8: (Placeholder) Integration of macroeconomic and alternative data can be added here.
    print("Training pipeline complete. Next steps: expand to multiple stocks, integrate external data, and explore further architecture enhancements.")

if __name__ == "__main__":
    main()
