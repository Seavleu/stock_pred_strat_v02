# """
# Final Pipeline Comparative Evaluation for Korean Stock Price Prediction

# This script compares two pipeline approaches:
#   Pipeline A: TSDataSampler-inspired LSTM pipeline with daily batching and per-stock normalization.
#   Pipeline B: Market-aware Transformer-based pipeline with dynamic feature selection and macroeconomic integration.

# Improvements include:
# 1️Dynamic feature selection using rolling-window SHAP (demo on one window).
# 2️Per-stock normalization and daily-batched sampling for progressive training.
# 3️Enhanced Transformer architecture with multi-layer market awareness.
# 4️Optimized training with adaptive learning rates, dropout, and batch normalization.

# The goal is to compare performance, generalization, and robustness in predicting stock prices for the Korean market.
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")  
# import matplotlib.pyplot as plt
# import seaborn as sns
# import optuna
# import shap
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# # ==========================
# # 1. Data Loading & Normalization
# # ==========================

# def load_stock_data(filepath):
#     """
#     Load pre-engineered stock data from CSV and apply per-stock normalization.
#     Assumes 'timestamp' and 'closing_price' columns exist.
#     """
#     df = pd.read_csv(filepath, parse_dates=['timestamp'])
#     df.sort_values("timestamp", inplace=True)
#     # Normalize numeric columns (per-stock normalization)
#     non_numeric = ["timestamp"]
#     feature_cols = [col for col in df.columns if col not in non_numeric]
#     scaler = StandardScaler()
#     df[feature_cols] = scaler.fit_transform(df[feature_cols])
#     return df

# # ==========================
# # 2. Dynamic Feature Selection (Rolling SHAP)
# # ==========================

# def dynamic_feature_selection(df, window_size=500, sample_size=200, mi_cutoff=0.1):
#     """
#     Perform dynamic feature selection using a rolling-window approach.
#     For demonstration, use the first window to compute mutual information and SHAP values.
#     Returns a list of selected features.
#     """
#     window_df = df.iloc[:window_size].copy()
#     X = window_df.drop(columns=["timestamp", "closing_price"])
#     y = window_df["closing_price"]

#     # Compute mutual information scores
#     mi_scores_array = mutual_info_regression(X, y)
#     mi_series = pd.Series(mi_scores_array, index=X.columns)
#     print("Rolling-window MI scores:")
#     print(mi_series.sort_values(ascending=False))
    
#     # Sample a subset for SHAP analysis
#     sample_size = min(sample_size, len(X))
#     X_sample = X.sample(n=sample_size, random_state=42)
    
#     # Use XGBoost for SHAP analysis (demo)
#     import xgboost as xgb
#     model = xgb.XGBRegressor(n_estimators=50, random_state=42)
#     model.fit(X_sample, y.loc[X_sample.index])
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_sample)
    
#     # Save SHAP summary plot
#     shap.summary_plot(shap_values, X_sample, show=False)
#     plt.title("Rolling-Window SHAP Summary")
#     plt.savefig("docs/rolling_window_shap.png")
#     plt.close()
#     print("SHAP summary plot saved as 'docs/rolling_window_shap.png'.")
    
#     # Select features with MI >= cutoff (for demo; further incorporate SHAP if desired)
#     selected_features = mi_series[mi_series >= mi_cutoff].index.tolist()
#     print("Dynamically selected features (MI >= {:.2f}):".format(mi_cutoff), selected_features)
#     return selected_features

# # ==========================
# # 3. Data Sampling & Augmentation (TSDataSampler Inspired)
# # ==========================

# def daily_batched_sampler(df, batch_size=64):
#     """
#     Simulate daily-batched sampling by grouping data by date.
#     Returns a DataFrameGroupBy object (for demo, assume one date per batch).
#     """
#     df['date'] = df['timestamp'].dt.date
#     grouped = df.groupby("date")
#     return grouped

# def augment_data(df, noise_std=0.01):
#     """
#     Apply Gaussian noise injection to augment data.
#     """
#     augmented = df.copy()
#     numeric_cols = augmented.select_dtypes(include=[np.number]).columns.tolist()
#     noise = np.random.normal(0, noise_std, size=augmented[numeric_cols].shape)
#     augmented[numeric_cols] = augmented[numeric_cols] + noise
#     return augmented

# # ==========================
# # 4. Dataset Definitions for LSTM and Transformer Pipelines
# # ==========================

# class TimeSeriesDataset(Dataset):
#     def __init__(self, df, seq_length=60, feature_columns=None, target_column="closing_price"):
#         """
#         Dataset for time-series forecasting.
#         """
#         self.seq_length = seq_length
#         self.feature_columns = feature_columns if feature_columns is not None else df.drop(columns=["timestamp", target_column]).columns.tolist()
#         self.data = df.sort_values("timestamp").reset_index(drop=True)
#         self.features = self.data[self.feature_columns].values
#         self.targets = self.data[target_column].values

#     def __len__(self):
#         return len(self.data) - self.seq_length

#     def __getitem__(self, idx):
#         x = self.features[idx: idx+self.seq_length]
#         y = self.targets[idx+self.seq_length]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# class TransformerStockDataset(Dataset):
#     def __init__(self, df, macro, seq_length=60, target_column="closing_price"):
#         """
#         Dataset that provides (x, y, macro) tuples.
#         """
#         self.seq_length = seq_length
#         self.data = df.sort_values("timestamp").reset_index(drop=True)
#         self.features = self.data.drop(columns=["timestamp", target_column]).values
#         self.targets = self.data[target_column].values
#         self.macro = macro  # macroeconomic data as numpy array (num_rows, num_macro)

#     def __len__(self):
#         return len(self.data) - self.seq_length

#     def __getitem__(self, idx):
#         x = self.features[idx: idx+self.seq_length]
#         y = self.targets[idx+self.seq_length]
#         # For macro, we use a simple average over the sequence window
#         macro_seq = self.macro[idx: idx+self.seq_length]
#         macro_avg = macro_seq.mean(axis=0)
#         return (torch.tensor(x, dtype=torch.float32),
#                 torch.tensor(y, dtype=torch.float32),
#                 torch.tensor(macro_avg, dtype=torch.float32))

# # ==========================
# # 5. Model Architectures
# # ==========================

# # -- Pipeline A: LSTM Model (TSDataSampler-inspired)
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
#         c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# # -- Pipeline B: Market-Aware Transformer Model
# class MarketAwareTransformer(nn.Module):
#     def __init__(self, feature_size, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.2, num_macro=5):
#         super(MarketAwareTransformer, self).__init__()
#         # Input projection for within-stock features
#         self.input_proj = nn.Linear(feature_size, d_model)
#         # Transformer encoder for within-stock trends
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         # Macro branch
#         self.macro_proj = nn.Linear(num_macro, d_model)
#         # Cross-stock attention (simulate with self-attention between stock and macro)
#         self.cross_stock_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
#         # Final prediction layer
#         self.fc_out = nn.Linear(d_model * 2, 1)
#         self.bn = nn.BatchNorm1d(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, macro_features):
#         # x: (batch, seq_length, feature_size)
#         x_proj = self.input_proj(x)  # (B, T, d_model)
#         x_proj = x_proj.permute(1, 0, 2)  # (T, B, d_model)
#         encoded = self.encoder(x_proj)  # (T, B, d_model)
#         stock_repr = encoded[-1]  # (B, d_model)
#         stock_repr = self.bn(stock_repr)
#         stock_repr = self.dropout(stock_repr)
#         macro_repr = self.macro_proj(macro_features)  # (B, d_model)
#         macro_repr_unsq = macro_repr.unsqueeze(0)  # (1, B, d_model)
#         attn_output, _ = self.cross_stock_attn(query=stock_repr.unsqueeze(0), key=macro_repr_unsq, value=macro_repr_unsq)
#         attn_output = attn_output.squeeze(0)  # (B, d_model)
#         combined = torch.cat((stock_repr, attn_output), dim=1)  # (B, 2*d_model)
#         out = self.fc_out(combined)
#         return out

# # ==========================
# # 6. Training Functions
# # ==========================
# def train_model(model, train_loader, num_epochs=20, lr=0.001):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0
#         for batch in train_loader:
#             # For Transformer pipeline, batch is (x, y, macro); for LSTM, it's (x, y)
#             if len(batch) == 3:
#                 x_batch, y_batch, macro_batch = batch
#                 x_batch, y_batch, macro_batch = x_batch.to(device), y_batch.to(device), macro_batch.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(x_batch, macro_batch)
#             else:
#                 x_batch, y_batch = batch
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(x_batch)
#             loss = criterion(outputs.squeeze(), y_batch)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * x_batch.size(0)
#         epoch_loss /= len(train_loader.dataset)
#         scheduler.step(epoch_loss)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
#     return model

# def evaluate_model(model, loader, pipeline="LSTM"):
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     preds, trues = [], []
#     with torch.no_grad():
#         for batch in loader:
#             if pipeline == "Transformer":
#                 x_batch, y_batch, macro_batch = batch
#                 x_batch, y_batch, macro_batch = x_batch.to(device), y_batch.to(device), macro_batch.to(device)
#                 outputs = model(x_batch, macro_batch).squeeze()
#             else:
#                 x_batch, y_batch = batch
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 outputs = model(x_batch).squeeze()
#             preds.extend(outputs.cpu().numpy())
#             trues.extend(y_batch.cpu().numpy())
#     mse = mean_squared_error(trues, preds)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(trues, preds)
#     mape = mean_absolute_percentage_error(trues, preds)
#     print(f"Evaluation -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
#     return mse, rmse, r2, mape

# # ==========================
# # 7. Main Comparative Pipeline
# # ==========================

# def main():
#     # Assume we have preprocessed/refined data for Samsung Electronics for Phase 1.
#     # File should be prepared with dynamic feature selection applied.
#     stock_file = "data/processed/refined_features_samsung.csv"
#     if not os.path.exists(stock_file) or os.path.getsize(stock_file) == 0:
#         sys.exit(f"Error: {stock_file} is missing or empty. Run the feature engineering and refinement pipeline for Samsung Electronics first.")
    
#     # Load and normalize data (per-stock normalization)
#     df = load_stock_data(stock_file)
#     print(f"Loaded {len(df)} rows for Samsung Electronics.")
    
#     # Dynamic Feature Selection using rolling-window SHAP
#     selected_features = dynamic_feature_selection(df, window_size=500, sample_size=200, mi_cutoff=0.1)
#     cols_to_keep = ["timestamp", "closing_price"] + selected_features
#     df = df[cols_to_keep].copy()
    
#     # For demonstration, augment data (optional)
#     df_aug = augment_data(df, noise_std=0.005)
    
#     # -------------------------
#     # Pipeline A: TSDataSampler-inspired LSTM
#     # -------------------------
#     # Create dataset for LSTM (daily-batched sampling simulated by using full data for now)
#     seq_length = 60  # can be tuned
#     feature_columns = df_aug.drop(columns=["timestamp", "closing_price"]).columns.tolist()
#     lstm_dataset = TimeSeriesDataset(df_aug, seq_length=seq_length, feature_columns=feature_columns)
#     lstm_loader = DataLoader(lstm_dataset, batch_size=64, shuffle=True)
    
#     # Instantiate and train LSTM model
#     input_size = len(feature_columns)
#     lstm_model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.3)
#     print("Training Pipeline A (LSTM)...")
#     lstm_model = train_model(lstm_model, lstm_loader, num_epochs=20, lr=0.001)
#     print("Evaluating Pipeline A (LSTM)...")
#     evaluate_model(lstm_model, lstm_loader, pipeline="LSTM")
    
#     # -------------------------
#     # Pipeline B: Enhanced Market-Aware Transformer
#     # -------------------------
#     # For Transformer pipeline, we need macroeconomic data.
#     # Here, we simulate macro data (e.g., FX rates, interest rates, market index, etc.) with 5 features.
#     num_macro = 5
#     macro_data = np.random.randn(len(df_aug), num_macro)  # Replace with real macro data if available
    
#     transformer_dataset = TransformerStockDataset(df_aug, macro=macro_data, seq_length=seq_length)
#     transformer_loader = DataLoader(transformer_dataset, batch_size=64, shuffle=True)
    
#     print("Training Pipeline B (Transformer)...")
#     transformer_model = MarketAwareTransformer(feature_size=input_size, num_macro=num_macro)
#     transformer_model = train_model(transformer_model, transformer_loader, num_epochs=20, lr=0.001)
#     print("Evaluating Pipeline B (Transformer)...")
#     evaluate_model(transformer_model, transformer_loader, pipeline="Transformer")
    
#     # -------------------------
#     # Comparative Evaluation
#     # -------------------------
#     print("Comparative evaluation complete. Compare performance metrics between LSTM and Transformer pipelines.")
#     print("Next steps: progressive training (adding more stocks), real macroeconomic integration, and validation on hold-out sets.")

# if __name__ == "__main__":
#     main()
