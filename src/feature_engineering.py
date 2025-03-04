"""
Feature Engineering Pipeline for KR Stock Market

- This script reads preprocessed stock data from:
    data/processed/combined_cleaned_stock_data.csv
- Computes advanced technical indicators, generates lag features, applies denoising(SG),
and integrates dynamic SA from Korean financial news sources 
- Finally, the enhanced dataset is saved to:
    data/processed/engineered_features.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler
from transformers import pipeline

#############################################
# Technical Indicator Functions
#############################################

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

#############################################
# Denoising & Lag Feature Generation
#############################################

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

#############################################
# Sentiment Analysis Functions
#############################################

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

#############################################
# Main Feature Engineering Pipeline
#############################################

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
# Main Execution Block
#############################################

def main():
    input_csv = "data/processed/combined_cleaned_stock_data.csv"
    output_csv = "data/processed/engineered_features.csv"
    
    # check that input file exists and is not empty
    if not os.path.exists(input_csv) or os.path.getsize(input_csv) == 0:
        raise FileNotFoundError(f"Error: {input_csv} is missing or empty. Please run the ingestion/cleaning pipeline first.")
    
    # read preprocessed data
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    
    # run the feature engineering pipeline
    engineered_df = feature_engineering_pipeline(df)
    
    # save the enhanced dataset
    engineered_df.to_csv(output_csv, index=False)
    print(f"Engineered features saved to {output_csv}")

if __name__ == "__main__":
    main()
