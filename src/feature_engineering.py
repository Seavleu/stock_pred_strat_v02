"""
Feature Engineering Pipeline for KR Stock Market

- This script reads cleaned stock data from:
    data/processed/korean_stock_data_cleaned.csv
- It computes advanced technical indicators and Alpha158-inspired features,
  generates lag features, applies denoising using the Savitzky–Golay filter, 
  and creates target columns for multi-day prediction:
      * next_day_close: closing price for the next day
      * day_after_next_close: closing price for the day after next
      * future_5day_close: closing price 5 days ahead
- Finally, the script scales numeric features using RobustScaler and saves the enhanced dataset to:
    data/processed/engineered_features.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression

#############################################
# Technical Indicator Functions
#############################################
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
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    return ma, upper_band, lower_band

def compute_moving_averages(series, windows=[5, 10, 20, 50]):
    ma_dict = {f"MA_{w}": series.rolling(window=w, min_periods=w).mean() for w in windows}
    return pd.DataFrame(ma_dict)

def compute_vwap(df, window=10):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window=window, min_periods=window).sum() / \
           df["volume"].rolling(window=window, min_periods=window).sum()
    return vwap

def compute_rate_of_change(series, window=1):
    return series.pct_change(periods=window) * 100

def compute_volatility(series, windows=[5, 10, 20]):
    vol_dict = {f"Volatility_{w}": series.rolling(window=w, min_periods=w).std() for w in windows}
    return pd.DataFrame(vol_dict)

def compute_momentum_features(series, windows=[5, 10, 20]):
    momentum = pd.DataFrame(index=series.index)
    for w in windows:
        momentum[f"Momentum_{w}"] = series.diff(w)
        momentum[f"Rolling_Max_{w}"] = series.rolling(window=w, min_periods=w).max()
        momentum[f"Rolling_Min_{w}"] = series.rolling(window=w, min_periods=w).min()
        momentum[f"Rank_{w}"] = series.rolling(window=w, min_periods=w).apply(lambda x: pd.Series(x).rank().iloc[-1])
    return momentum

#############################################
# Denoising & Lag Feature Generation
#############################################
def apply_savgol_filter(series, window_length=11, polyorder=2):
    if len(series) < window_length:
        window_length = len(series) // 2 * 2 + 1
    return savgol_filter(series, window_length=window_length, polyorder=polyorder)

def create_lag_features(df, column, lags=[1, 3, 5, 10, 20]):
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

#############################################
# Alpha158-inspired & Market-Wide Feature Functions
#############################################
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
        print("volume not found. Skipping VWAP variations.")
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
    # Compute Alpha158-inspired features
    df = compute_regression_features(df, window=30)
    df = compute_kbar_features(df)
    df = compute_enhanced_lag_features(df, "close", windows=[3,5,10,20])
    df = compute_vwap_variations(df)
    df = integrate_macroeconomic_data(df)
    df = compute_market_indicators(df)
    return df

#############################################
# Main Feature Engineering Pipeline
#############################################
def feature_engineering_pipeline(df):
    df = df.copy()
    # Ensure proper datetime format and sort by date
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    
    # Compute standard technical indicators based on close price
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
    
    # Apply denoising to the close price using Savitzky-Golay filter
    df["Close_Denoised"] = apply_savgol_filter(df["close"])
    
    # Create lag features for close price
    df = create_lag_features(df, "close", lags=[1,3,5,10,20])
    
    # Compute Alpha158-inspired features and integrate market-wide indicators
    df = compute_alpha158_features(df)
    
    # Create target columns for multi-day forecasting:
    # next_day_close: closing price shifted by -1
    # day_after_next_close: closing price shifted by -2
    # future_5day_close: closing price shifted by -5
    df["next_day_close"] = df["close"].shift(-1)
    df["day_after_next_close"] = df["close"].shift(-2)
    df["future_5day_close"] = df["close"].shift(-5)
    
    # Drop rows with NaNs from shifting operations
    # df.dropna(inplace=True)
    
     # Replace infinite values and drop rows with NaN before scaling
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Scale all numeric features using RobustScaler
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def main():
    input_csv = "data/processed/korean_stock_data_cleaned.csv"
    output_csv = "data/processed/engineered_features.csv"
    
    if not os.path.exists(input_csv) or os.path.getsize(input_csv) == 0:
        raise FileNotFoundError(f"Error: {input_csv} is missing or empty. Run the cleaning pipeline first.")
    
    df = pd.read_csv(input_csv, parse_dates=["date"])
    engineered_df = feature_engineering_pipeline(df)
    engineered_df.to_csv(output_csv, index=False)
    print(f"Engineered features saved to {output_csv}")

if __name__ == "__main__":
    main()
