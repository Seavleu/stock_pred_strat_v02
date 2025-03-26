import os
import sys
import numpy as np
import pandas as pd
import pickle
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import logging

# ----------------------------
# Setup logging (or import your logging module)
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Data Loading (assume cleaned data already exists)
# ----------------------------
def load_cleaned_data(path="data/processed/korean_stock_data_cleaned.csv"):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        sys.exit(f"Error: {path} is missing or empty. Run the data cleaning pipeline first.")
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} rows from {path}.")
    return df

# ----------------------------
# Feature Engineering Stage
# ----------------------------
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return ma, ma + num_std * std, ma - num_std * std

def compute_moving_averages(series, windows=[5, 10, 20, 50]):
    ma_dict = {f"MA_{w}": series.rolling(window=w, min_periods=w).mean() for w in windows}
    return pd.DataFrame(ma_dict)

def apply_savgol_filter(series, window_length=11, polyorder=2):
    if len(series) < window_length:
        window_length = len(series) // 2 * 2 + 1
    return savgol_filter(series, window_length=window_length, polyorder=polyorder)

def create_lag_features(df, column, lags=[1, 3, 5, 10, 20]):
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

# Alpha158-inspired features (similar to your existing functions)
def compute_returns(series):
    return series.pct_change()

def compute_regression_features(df, window=30):
    df["stock_return"] = compute_returns(df["close"])
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
    low_col = "low" if "low" in df.columns else "Rolling_Min_5"
    df["KMID"] = (df["high"] + df[low_col]) / 2
    df["KLEN"] = df["high"] - df[low_col]
    df["KSFT"] = df["close"] - df["KMID"]
    return df

def compute_enhanced_lag_features(df, column, windows=[3, 5, 10, 20]):
    for w in windows:
        df[f"{column}_lag_mean_{w}"] = df[column].rolling(window=w, min_periods=1).mean().shift(1)
        df[f"{column}_lag_std_{w}"] = df[column].rolling(window=w, min_periods=1).std().shift(1)
    return df

def compute_vwap_variations(df):
    if "volume" not in df.columns:
        logger.warning("volume not found. Skipping VWAP variations.")
        return df
    low_col = "low" if "low" in df.columns else "Rolling_Min_5"
    df["typical_price"] = (df["high"] + df[low_col] + df["close"]) / 3
    df["vwap_typical"] = (df["typical_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    if "open" in df.columns:
        df["ohlc4_price"] = (df["open"] + df["high"] + df[low_col] + df["close"]) / 4
        df["vwap_ohlc4"] = (df["ohlc4_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["hlc3_price"] = (df["high"] + df[low_col] + df["close"]) / 3
    df["vwap_hlc3"] = (df["hlc3_price"] * df["volume"]).cumsum() / df["volume"].cumsum()
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
    logger.info("Starting Alpha158 features computation")
    df = compute_regression_features(df, window=30)
    df = compute_kbar_features(df)
    df = compute_enhanced_lag_features(df, "close", windows=[3,5,10,20])
    df = compute_vwap_variations(df)
    df = integrate_macroeconomic_data(df)
    df = compute_market_indicators(df)
    logger.info("Alpha158 features computation completed")
    return df

def feature_engineering_pipeline(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    
    # Standard technical indicators
    df["RSI"] = compute_rsi(df["close"])
    macd_line, signal_line, macd_hist = compute_macd(df["close"])
    df["MACD_Line"] = macd_line
    df["Signal_Line"] = signal_line
    df["MACD_Hist"] = macd_hist
    bb_ma, bb_upper, bb_lower = compute_bollinger_bands(df["close"])
    df["BB_MA"] = bb_ma
    df["BB_Upper"] = bb_upper
    df["BB_Lower"] = bb_lower
    ma_df = compute_moving_averages(df["close"])
    df = pd.concat([df, ma_df], axis=1)
    
    df["VWAP"] = compute_vwap(df, window=10)
    df["ROC"] = compute_rate_of_change(df["close"], window=1)
    vol_df = compute_volatility(df["close"], windows=[5,10,20])
    df = pd.concat([df, vol_df], axis=1)
    momentum_df = compute_momentum_features(df["close"], windows=[5,10,20])
    df = pd.concat([df, momentum_df], axis=1)
    
    df["Close_Denoised"] = apply_savgol_filter(df["close"])
    df = create_lag_features(df, "close", lags=[1,3,5,10,20])
    
    df = compute_alpha158_features(df)
    
    # Create target cols for multi-day forecasting
    df["next_day_close"] = df["close"].shift(-1)
    df["day_after_next_close"] = df["close"].shift(-2)
    df["future_5day_close"] = df["close"].shift(-5)
    
    # Remove rows with missing or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Scale numeric features with RobustScaler
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

# ----------------------------
# Feature Refinement Stage
# ----------------------------
def remove_correlated_features(df, features, threshold=0.99):
    corr_matrix = df[features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            if upper_triangle.loc[row, col] > threshold:
                to_drop.add(col)
    refined_features = [f for f in features if f not in to_drop]
    logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    return refined_features

def remove_low_value_features(df, features, mi_scores, mi_cutoff=0.1, shap_importances=None, shap_cutoff=None):
    keep_by_mi = [f for f in features if mi_scores.get(f, 0) >= mi_cutoff]
    dropped_by_mi = set(features) - set(keep_by_mi)
    logger.info(f"Dropped {len(dropped_by_mi)} features with MI < {mi_cutoff}: {dropped_by_mi}")
    if shap_importances is not None and shap_cutoff is not None:
        keep_by_shap = [f for f in keep_by_mi if shap_importances.get(f, 0) >= shap_cutoff]
        dropped_by_shap = set(keep_by_mi) - set(keep_by_shap)
        logger.info(f"Dropped {len(dropped_by_shap)} features with SHAP < {shap_cutoff}: {dropped_by_shap}")
        final_features = keep_by_shap
    else:
        final_features = keep_by_mi
    return final_features

def add_future_return_targets(df):
    epsilon = 1e-10
    df = df.copy()
    df['next_day_return'] = df['close'].shift(-1) / (df['close'] + epsilon) - 1
    df['day_after_next_return'] = df['close'].shift(-2) / (df['close'] + epsilon) - 1
    df['future_avg_return'] = (df['next_day_return'] + df['day_after_next_return']) / 2
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['future_avg_return'], inplace=True)
    return df

def feature_refinement_pipeline(df):
    df = add_future_return_targets(df)
    all_cols = df.columns.tolist()
    non_feature_cols = ["date", "close", "next_day_return", "day_after_next_return", "future_avg_return"]
    candidate_features = [col for col in all_cols if col not in non_feature_cols]
    feature_cols = [col for col in candidate_features if pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols]
    y = df["future_avg_return"]
    mi_scores_array = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores_array, index=X.columns).to_dict()
    for feature, score in mi_scores.items():
        logger.info(f"MI score for {feature}: {score:.4f}")
    
    refined_features = remove_correlated_features(df, feature_cols, threshold=0.99)
    refined_features = remove_low_value_features(df, refined_features, mi_scores, mi_cutoff=0.1)
    logger.info(f"Final refined features: {refined_features}")
    
    keep_cols = ["date", "future_avg_return"] + refined_features
    refined_df = df[keep_cols].copy()
    return refined_df

# ----------------------------
# Dynamic Feature Scaling Stage
# ----------------------------
def scale_features(df, returns_features, technical_features):
    df_scaled = df.copy()
    if returns_features:
        scaler_mm = MinMaxScaler(feature_range=(-1, 1))
        df_scaled[returns_features] = scaler_mm.fit_transform(df_scaled[returns_features])
    if technical_features:
        scaler_std = StandardScaler()
        df_scaled[technical_features] = scaler_std.fit_transform(df_scaled[technical_features])
    return df_scaled

# ----------------------------
# Evaluation Metrics
# ----------------------------
def calculate_ic_ric(y_true, y_pred):
    ic, _ = pearsonr(y_true, y_pred)
    ric, _ = spearmanr(y_true, y_pred)
    return ic, ric

# ----------------------------
# Model Components and Training (LSTM)
# ----------------------------
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
        logger.info(f"Final Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
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
        raise ValueError("Evaluation data contains NaN values.")
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    ic, ric = calculate_ic_ric(actuals, predictions)
    logger.info(f"Evaluation Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    logger.info(f"Information Coefficient (IC): {ic:.4f}, Rank IC (RIC): {ric:.4f}")
    return mse, rmse, r2, mape, ic, ric

def train_val_test_split_ts(df, train_size=0.8, val_size=0.1, test_size=0.1):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df

# ----------------------------
# Main Pipeline Function
# ----------------------------
def main():
    # Load engineered features (output from feature_engineering.py)
    input_csv = "data/processed/engineered_features.csv"
    if not os.path.exists(input_csv) or os.path.getsize(input_csv) == 0:
        sys.exit("Error: Engineered features CSV is missing or empty. Run feature_engineering.py first.")
    df = pd.read_csv(input_csv, parse_dates=["date"])
    logger.info(f"Loaded {len(df)} rows from {input_csv}.")

    # For single-stock training, filter by company if necessary
    if "company" in df.columns:
        companies = df["company"].unique()
        logger.info(f"Found companies: {companies}")
        selected_company = companies[0]
        logger.info(f"Using data for company: {selected_company}")
        df = df[df["company"] == selected_company].copy()

    # Ensure target column exists; add if missing
    if "future_avg_return" not in df.columns:
        from src.feature_engineering.feature_refinement import add_future_return_targets
        df = add_future_return_targets(df)
        logger.info("Computed future_avg_return target column.")

    # Determine feature cols (exclude date and target)
    all_cols = df.columns.tolist()
    non_feature_cols = ["date", "future_avg_return"]
    feature_cols = [col for col in all_cols if col not in non_feature_cols]

    # Dynamic feature scaling: for single-stock, we can scale globally
    returns_features = []  # adjust if you have any returns-based features
    technical_features = feature_cols
    df_scaled = scale_features(df, returns_features, technical_features)
    scaled_csv = "data/processed/engineered_features_scaled.csv"
    df_scaled.to_csv(scaled_csv, index=False)
    logger.info(f"Scaled dataset saved to {scaled_csv}")

    # Prepare dataset and split for training
    train_df, val_df, test_df = train_val_test_split_ts(df_scaled, train_size=0.8, val_size=0.1, test_size=0.1)
    logger.info(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}, Test shape: {test_df.shape}")

    # For training, we use the scaled dataset (single-stock)
    train_dataset = StockDataset(train_df, seq_length=30, feature_columns=feature_cols, target_column="future_avg_return")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    input_size = len(feature_cols)

    # Hyperparameter tuning using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size), n_trials=10)
    logger.info(f"Best hyperparameters: {study.best_params}")
    best_params = study.best_params

    # Train final model
    final_model = train_final_model(train_loader, input_size, best_params, num_epochs=20)
    evaluate_model(final_model, train_loader)

    logger.info("Enhanced evaluation pipeline complete. The model now uses dynamic feature scaling and is evaluated with IC & RIC metrics.")

if __name__ == "__main__":
    main()
