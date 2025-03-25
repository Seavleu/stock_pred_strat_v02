'''
This pipeline is scalable for both single-stock and multi-stock. 
'''
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def load_cleaned_data(self):
        """Load the cleaned dataset."""
        return pd.read_csv(self.config['paths']['cleaned_data'])

    def compute_technical_indicators(self, df):
        # Moving Averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        
        # RSI Calculation (14-period)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # add epsilon to avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD: EMA12 and EMA26
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands (20-period)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        return df

    def compute_alpha158_features(self, df):
        # Simple proxies for alpha158-inspired features
        df['beta'] = df['close'].pct_change().rolling(window=20).std()
        df['rsqr'] = df['close'].pct_change().rolling(window=20).var()
        df['resi'] = df['close'].rolling(window=20).mean() - df['close']
        return df

    def compute_candlestick_features(self, df):
        df['kmid'] = (df['high'] + df['low']) / 2
        df['klen'] = df['high'] - df['low']
        df['ksft'] = df['close'] - df['open']
        return df

    def compute_lag_features(self, df):
        # Create lags 1 to 5 for close prices
        for lag in range(1, 6):
            df[f'lag_{lag}'] = df['close'].shift(lag)
        df['lag_mean'] = df[[f'lag_{lag}' for lag in range(1, 6)]].mean(axis=1)
        df['lag_std'] = df[[f'lag_{lag}' for lag in range(1, 6)]].std(axis=1)
        return df

    def compute_additional_features(self, df):
        # VWAP: simplified calculation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3) / df['volume']
        df['roc'] = df['close'].pct_change(periods=10)
        df['volatility'] = df['close'].rolling(window=10).std()
        df['momentum'] = df['close'] - df['close'].shift(10)
        return df

    def denoise_close(self, df):
        # Apply Savitzky-Golay filter to denoise the 'close' price if sufficient data
        if len(df) >= 7:
            df['close_denoised'] = savgol_filter(df['close'], window_length=7, polyorder=2)
        else:
            df['close_denoised'] = df['close']
        return df

    def create_target_columns(self, df):
        # Create multi-day prediction targets
        df['next_day_close'] = df['close'].shift(-1)
        df['day_after_next_close'] = df['close'].shift(-2)
        df['future_5day_close'] = df['close'].shift(-5)
        return df

    def save_engineered_features(self, df):
        """Save the engineered features to a CSV file."""
        path = self.config['paths']['engineered_features']
        df.to_csv(path, index=False)
        return df

    def run_feature_engineering(self):
        df = self.load_cleaned_data()
        df = self.compute_technical_indicators(df)
        df = self.compute_alpha158_features(df)
        df = self.compute_candlestick_features(df)
        df = self.compute_lag_features(df)
        df = self.compute_additional_features(df)
        df = self.denoise_close(df)
        df = self.create_target_columns(df)
        df = self.save_engineered_features(df)
        return df

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    fe = FeatureEngineer(config)
    engineered_df = fe.run_feature_engineering()
    print("Feature engineering completed. Engineered features saved to:", config['paths']['engineered_features'])
