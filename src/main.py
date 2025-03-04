import logging
import os
import glob
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline

# --- Log Tracker ---
logging.basicConfig(
    filename="training.log",
    filemode="a",   
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#############################################
# 1️⃣ Feature Engineering & Data Preprocessing
#############################################

# --- Technical Indicator Functions ---

def compute_rsi(series, window=14):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Compute MACD: returns MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_bollinger_bands(series, window=20, num_std=2):
    """Compute Bollinger Bands: moving average, upper and lower bands."""
    ma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    return ma, upper_band, lower_band

def compute_moving_averages(series, windows=[5, 10, 20, 50]):
    """Compute multiple simple moving averages."""
    ma_dict = {f"MA_{w}": series.rolling(window=w, min_periods=w).mean() for w in windows}
    return pd.DataFrame(ma_dict)

def compute_vwap(df, window=10):
    """
    Compute the Volume-Weighted Average Price (VWAP) over a rolling window.
    Typical price is approximated as (High + Low + Close) / 3.
    """
    typical_price = (df['highest_price'] + df['lowest_price'] + df['closing_price']) / 3
    vwap = (typical_price * df['trading_volume']).rolling(window=window, min_periods=window).sum() / \
           df['trading_volume'].rolling(window=window, min_periods=window).sum()
    return vwap

def compute_rate_of_change(series, window=1):
    """
    Compute the Rate of Change (ROC) as percentage change over a given window.
    """
    roc = series.pct_change(periods=window) * 100
    return roc

def compute_volatility(series, windows=[5, 10, 20]):
    """Compute rolling volatility (standard deviation) for given window sizes."""
    vol_dict = {f"Volatility_{w}": series.rolling(window=w, min_periods=w).std() for w in windows}
    return pd.DataFrame(vol_dict)

def compute_momentum_features(series, windows=[5, 10, 20]):
    """
    Compute momentum-based features:
      - Price trend: difference between current and lagged price.
      - Rolling max/min over window.
      - Rank of current price in the rolling window.
    """
    momentum_features = pd.DataFrame(index=series.index)
    for w in windows:
        momentum_features[f"Momentum_{w}"] = series.diff(w)
        momentum_features[f"Rolling_Max_{w}"] = series.rolling(window=w, min_periods=w).max()
        momentum_features[f"Rolling_Min_{w}"] = series.rolling(window=w, min_periods=w).min()
        momentum_features[f"Rank_{w}"] = series.rolling(window=w, min_periods=w).apply(lambda x: pd.Series(x).rank().iloc[-1])
    return momentum_features

# --- Sentiment Analysis with Korean NLP Model ---
# Denoising & Lag Feature Generation
def apply_savgol_filter(series, window_length=11, polyorder=2):
    """Apply Savitzky-Golay filter for smoothing a time series."""
    # Ensure window_length is odd and does not exceed series length
    if len(series) < window_length:
        window_length = len(series) // 2 * 2 + 1
    return savgol_filter(series, window_length=window_length, polyorder=polyorder)

def create_lag_features(df, column, lags=[1, 3, 5, 10, 20]):
    """Generate lag features for a given column."""
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

def fetch_korean_news():
    """
    Fetch real-time Korean financial news headlines.
    (This is a stub; integrate actual API calls to Naver Finance, Daum Finance, or Investing.com Korea.)
    """
    headlines = [
        "삼성전자, 글로벌 반도체 경쟁력 강화로 주가 상승 기대",
        "LG화학, 친환경 사업 확대로 투자 심리 개선",
        "현대자동차, 전기차 수요 증가에 따른 실적 호조 전망"
    ]
    return headlines

def analyze_sentiment(headlines):
    """
    Analyze sentiment using a Korean NLP model (KoBERT/KoElectra).
    Returns a dynamic sentiment score.
    """
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="monologg/koelectra-base-v3-discriminator",
        tokenizer="monologg/koelectra-base-v3-discriminator"
    )
    scores = []
    for headline in headlines:
        result = sentiment_pipeline(headline)[0]
        # convert sentiment label to numeric score: positive = score, negative = -score.
        score = result['score'] if result['label'].upper() == "POSITIVE" else -result['score']
        scores.append(score)
    return np.mean(scores) if scores else 0

def feature_engineering_pipeline(df):
    """
    Build an advanced feature engineering pipeline:
      - Compute technical indicators.
      - Apply Savitzky-Golay filtering.
      - Generate lag features.
      - Integrate real-time sentiment analysis.
      - Handle missing data and outliers.
      - Apply robust scaling.
    
    Args:
        df (DataFrame): Raw, cleaned stock data with columns such as:
                        ['timestamp', 'opening_price', 'highest_price', 'lowest_price',
                         'closing_price', 'trading_volume', ...]
    
    Returns:
        DataFrame: Enhanced dataset ready for LSTM/Seq2Seq models.
    """
    df = df.copy()
    
    # ensure proper datetime format and sort by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values("timestamp", inplace=True)
    
    # technical indicators on 'closing_price'
    df['RSI'] = compute_rsi(df['closing_price'])
    macd_line, signal_line, macd_hist = compute_macd(df['closing_price'])
    df['MACD_Line'] = macd_line
    df['Signal_Line'] = signal_line
    df['MACD_Hist'] = macd_hist
    bb_ma, bb_upper, bb_lower = compute_bollinger_bands(df['closing_price'])
    df['BB_MA'] = bb_ma
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    ma_df = compute_moving_averages(df['closing_price'])
    df = pd.concat([df, ma_df], axis=1)
    
    # Additional Technical Indicators
    df['VWAP'] = compute_vwap(df, window=10)
    df['ROC'] = compute_rate_of_change(df['closing_price'], window=1)
    volatility_df = compute_volatility(df['closing_price'], windows=[5, 10, 20])
    df = pd.concat([df, volatility_df], axis=1)
    momentum_df = compute_momentum_features(df['closing_price'], windows=[5, 10, 20])
    df = pd.concat([df, momentum_df], axis=1)
    
    # denoising: apply Savitzky-Golay filter to closing price
    df['Close_Denoised'] = apply_savgol_filter(df['closing_price'])
    
    # lag features: Create lag features for closing price
    df = create_lag_features(df, "closing_price", lags=[1, 3, 5, 10, 20])
    
    # SA: Fetch headlines and compute dynamic sentiment score
    headlines = fetch_korean_news()
    df['Sentiment'] = analyze_sentiment(headlines)
    
    # handle missing data: forward fill then backward fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # outlier Removal: IQR clipping for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
    
    # feature Scaling: RobustScaler on selected numeric features
    features_to_scale = numeric_cols.copy()
    scaler = RobustScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # drop any remaining NaN values (rolling calcs)
    df.dropna(inplace=True)
    
    return df

#############################################
# 2️⃣ Dataset & DataLoader for Time-Series Forecasting
#############################################

class StockDataset(Dataset):
    def __init__(self, csv_file, seq_length=30):
        """
        Args:
            csv_file (str): Path to the engineered features CSV.
            seq_length (int): Number of time steps per sequence.
        """
        self.data = pd.read_csv(csv_file, parse_dates=['timestamp'])
        self.data.sort_values("timestamp", inplace=True)
        # Drop timestamp column for modeling
        self.features = self.data.drop(columns=["timestamp"]).values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_length]
        # Here, we predict the next time step's closing price (adjust index as needed)
        # Assuming 'closing_price' is the 4th column among numeric features.
        # Adjust target index if your CSV column order differs.
        target_index = list(self.data.columns).index("closing_price") - 1  # because timestamp is dropped
        y = self.features[idx + self.seq_length, target_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#############################################
# 3️⃣ Seq2Seq LSTM with Bahdanau Attention
#############################################

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# --- Bahdanau Attention ---
'''
Coventional encoder-decoder architectures for machine translation encoded every source sentnece into a fixed-length vector
regardless its length, from whcih the decoder would then generate a translation. This made it difficualt for the NN to cope
with long sentences, essentially resulting in a performance bottleneck.

The Bahdanau attention was supposed to address the performance bottleneck.
'''
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_size] (decoder hidden state)
        # encoder_outputs: [batch, seq_len, hidden_size]
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, seq_len, hidden_size]
        energy = energy.transpose(1, 2)  # [batch, hidden_size, seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch, 1, hidden_size]
        attention = torch.bmm(v, energy).squeeze(1)  # [batch, seq_len]
        return torch.softmax(attention, dim=1)

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size + output_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attention = BahdanauAttention(hidden_size)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch, output_size] (previous output)
        # hidden: [num_layers, batch, hidden_size]
        # encoder_outputs: [batch, seq_len, hidden_size]
        # Calculate attention weights
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, seq_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, hidden_size]
        
        # Concatenate input and context
        input = input.unsqueeze(1)  # [batch, 1, output_size]
        lstm_input = torch.cat((input, context), dim=2)  # [batch, 1, hidden_size+output_size]
        
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((outputs.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, hidden, cell

# --- Seq2Seq Model ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, seq_length):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.seq_length = seq_length

    def forward(self, x, target_len):
        batch_size = x.size(0)
        # Encoder forward pass
        encoder_outputs, hidden, cell = self.encoder(x)
        # First decoder input (e.g., last closing price from the sequence)
        input_decoder = x[:, -1, :1]  # using the first feature as a proxy; adjust as needed
        
        outputs = []
        for t in range(target_len):
            output, hidden, cell = self.decoder(input_decoder, hidden, cell, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            input_decoder = output  # feeding output back as input
        outputs = torch.cat(outputs, dim=1)
        return outputs

#############################################
# 4️⃣ Hyperparameter Tuning & Training Optimization with Optuna
#############################################

def objective(trial):
    # Hyperparameters to tune
    seq_length = trial.suggest_int("seq_length", 20, 50)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_size = trial.suggest_int("hidden_size", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = 10  # for tuning, keep epochs lower

    # Load dataset
    data_path = "data/processed/engineered_features.csv"
    dataset = StockDataset(data_path, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Determine input size from CSV (drop timestamp)
    df = pd.read_csv(data_path)
    input_size = df.drop(columns=["timestamp"]).shape[1]
    
    # Define model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
    decoder = Decoder(output_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model = Seq2Seq(encoder, decoder, device, seq_length).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Predict one step ahead (target_len = 1)
            outputs = model(x_batch, target_len=1).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return epoch_loss

#############################################
# 5️⃣ Model Evaluation & Trade Signal Metrics
#############################################

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch, target_len=1).squeeze(1)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    logging.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}")
    # Placeholder for trading metrics (Sharpe Ratio, Maximum Drawdown)
    return mse, rmse, r2, mape

#############################################
# 6️⃣ Optional: Reinforcement Learning for Trade Execution
#############################################
# Stub: Integrate an RL strategy (PPO/DQN) using stable-baselines3 and a backtesting library (Backtrader/Zipline)
# This section can be expanded once the supervised forecasting model is validated.

#############################################
# 7️⃣ Main Training & Deployment Pipeline
#############################################

def main():
    # --- Feature Engineering Step ---
    # Load cleaned data (ensure your data pipeline has created this file)
    combined_csv = "data/processed/combined_cleaned_stock_data.csv"
    if not os.path.exists(combined_csv) or os.path.getsize(combined_csv) == 0:
        sys.exit(f"Error: {combined_csv} is missing or empty. Run the ingestion/cleaning pipeline first.")
    
    raw_df = pd.read_csv(combined_csv, parse_dates=['timestamp'])
    engineered_df = feature_engineering_pipeline(raw_df)
    engineered_csv = "data/processed/engineered_features.csv"
    engineered_df.to_csv(engineered_csv, index=False)
    logging.info(f"Engineered features saved to {engineered_csv}")
    
    # --- Hyperparameter Tuning ---
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    logging.info("Best hyperparameters:", study.best_params)
    
    # --- Train Final Model with Best Hyperparameters ---
    best_params = study.best_params
    seq_length = best_params["seq_length"]
    batch_size = best_params["batch_size"]
    hidden_size = best_params["hidden_size"]
    num_layers = best_params["num_layers"]
    dropout = best_params["dropout"]
    learning_rate = best_params["learning_rate"]
    num_epochs = 20  # Increase epochs for final training
    
    dataset = StockDataset(engineered_csv, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    df = pd.read_csv(engineered_csv)
    input_size = df.drop(columns=["timestamp"]).shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
    decoder = Decoder(output_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model = Seq2Seq(encoder, decoder, device, seq_length).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch, target_len=1).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(dataset)
        scheduler.step(epoch_loss)
        logging.info(f"Final Training - Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        # Early Stopping could be added here based on validation metrics
    
    # Save the final model
    model_save_path = "models/seq2seq_lstm_attention.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Final model saved to {model_save_path}")
    
    # --- Model Evaluation ---
    # For evaluation, use a held-out validation set or cross-validation.
    # Here, we use the training set for demonstration.
    evaluate_model(model, dataloader, device)
    
    # --- Deployment Placeholder ---
    # Build a FastAPI endpoint and containerize with Docker as needed.
    print("Deployment pipeline can now be set up, do it later;D")

if __name__ == "__main__":
    main()
