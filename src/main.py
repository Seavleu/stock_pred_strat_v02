# src/main.py

import pandas as pd
import numpy as np
import joblib
import yaml
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import savgol_filter

# ----------------------------
# Preprocessing Module
# ----------------------------
class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def load_raw_data(self):
        raw_data_path = self.config['data_paths']['raw_data']
        df = pd.read_csv(raw_data_path)
        return df

    def clean_data(self, df):
        # Remove rows with missing or "unknown" company names
        if 'CompanyName' in df.columns:
            df = df[df['CompanyName'].notna()]
            df = df[~df['CompanyName'].str.contains("unknown", case=False)]
            # Standardize company names (remove common suffixes)
            df['CompanyName'] = df['CompanyName'].apply(
                lambda x: re.sub(r"\s*(Co\.?,?\s*Ltd\.?)$", "", x))
        # Apply quality filters: e.g., minimum average volume filter
        if 'Volume' in df.columns:
            df = df[df['Volume'] > self.config['quality_filters']['min_avg_volume']]
        # Clip outliers using the IQR method for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def normalize_numeric(self, df, fit=True):
        # Normalize selected numeric cols using MinMaxScaler
        numeric_cols = self.config['scaling']['numeric_columns']
        if fit:
            df[numeric_cols] = self.minmax_scaler.fit_transform(df[numeric_cols])
            joblib.dump(self.minmax_scaler, self.config['paths']['minmax_scaler'])
        else:
            self.minmax_scaler = joblib.load(self.config['paths']['minmax_scaler'])
            df[numeric_cols] = self.minmax_scaler.transform(df[numeric_cols])
        return df

    def save_cleaned_data(self, df):
        processed_path = self.config['paths']['cleaned_data']
        df.to_csv(processed_path, index=False)
        # Optionally, split per company using 'Ticker' if available
        if 'Ticker' in df.columns:
            for ticker, group in df.groupby('Ticker'):
                group.to_csv(f"{self.config['paths']['extracted_dir']}/{ticker}.csv", index=False)
        return df

# ----------------------------
# Feature Engineering Module
# ----------------------------
class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def compute_technical_indicators(self, df):
        # Moving Averages (MA_5, MA_10)
        if 'Close' in df.columns:
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
        # RSI Calculation (simplified)
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + roll_up / roll_down))
        # MACD Calculation
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        return df

    def compute_alpha158_features(self, df):
        # Dummy implementations for regression-based features
        df['BETA'] = df['Close'].pct_change().rolling(window=20).std()
        df['RSQR'] = df['Close'].pct_change().rolling(window=20).var()
        df['RESI'] = df['Close'].rolling(window=20).mean() - df['Close']
        return df

    def compute_candlestick_features(self, df):
        # K-bar features: KMID, KLEN, KSFT
        df['KMID'] = (df['High'] + df['Low']) / 2
        df['KLEN'] = df['High'] - df['Low']
        df['KSFT'] = df['Close'] - df['Open']
        return df

    def compute_lag_features(self, df):
        # Enhanced lag features: mean and standard deviation of lagged close prices
        for lag in range(1, 6):
            df[f'lag_{lag}'] = df['Close'].shift(lag)
        df['lag_mean'] = df[[f'lag_{lag}' for lag in range(1, 6)]].mean(axis=1)
        df['lag_std'] = df[[f'lag_{lag}' for lag in range(1, 6)]].std(axis=1)
        return df

    def compute_additional_features(self, df):
        # VWAP, ROC, volatility, and momentum-based features (simplified)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3) / df['Volume']
        df['ROC'] = df['Close'].pct_change(periods=10)
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        return df

    def denoise_close(self, df):
        # Apply Savitzky–Golay filter to denoise 'Close' price
        df['Close_denoised'] = savgol_filter(df['Close'], window_length=7, polyorder=2)
        return df

    def create_target_columns(self, df):
        # Create multi-day prediction targets
        df['next_day_close'] = df['Close'].shift(-1)
        df['day_after_next_close'] = df['Close'].shift(-2)
        df['future_5day_close'] = df['Close'].shift(-5)
        return df

    def save_engineered_features(self, df):
        path = self.config['paths']['engineered_features']
        df.to_csv(path, index=False)
        return df

# ----------------------------
# Feature Refinement Module
# ----------------------------
class FeatureRefiner:
    def __init__(self, config):
        self.config = config

    def drop_highly_correlated(self, df):
        # Remove features with correlation above a specified threshold (e.g., 0.99)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.config['refinement']['corr_threshold'])]
        df = df.drop(columns=to_drop)
        return df

    def drop_low_mi_features(self, df, target_col):
        # Compute mutual information and drop features with MI below the threshold
        features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        mi = mutual_info_regression(features.fillna(0), df[target_col].fillna(0))
        low_mi_cols = features.columns[mi < self.config['refinement']['mi_threshold']].tolist()
        df = df.drop(columns=low_mi_cols)
        return df

    def transform_target(self, df):
        # Create a target transformation (e.g., future average return)
        df['future_avg_return'] = ((df['next_day_close'] + df['day_after_next_close']) / 2) - df['Close']
        return df

    def save_refined_features(self, df):
        path = self.config['paths']['refined_features']
        df.to_csv(path, index=False)
        return df

# ----------------------------
# Dynamic Feature Selection Module
# ----------------------------
class DynamicFeatureSelector:
    def __init__(self, config):
        self.config = config

    def rolling_shap_analysis(self, df):
        # Placeholder for rolling-window SHAP analysis.
        # For each window (e.g., 200 rows with step 50), an XGBoost model is trained and SHAP values are aggregated.
        # Here, we simply select the top 50% of features arbitrarily.
        features = df.columns.tolist()
        selected_features = features[: len(features)//2]
        return df[selected_features]

    def save_dynamic_features(self, df):
        path = self.config['paths']['dynamic_features']
        df.to_csv(path, index=False)
        return df

# ----------------------------
# Enhanced Evaluation & Scaling Module
# ----------------------------
class EvaluatorScaler:
    def __init__(self, config):
        self.config = config
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def dynamic_scaling(self, df):
        # Normalize returns-based features using MinMax scaling to [-1, 1]
        for col in self.config['scaling']['returns_columns']:
            if col in df.columns:
                df[[col]] = self.minmax_scaler.fit_transform(df[[col]])
        # Standardize technical indicators using Z-score normalization
        for col in self.config['scaling']['technical_columns']:
            if col in df.columns:
                df[[col]] = self.std_scaler.fit_transform(df[[col]])
        joblib.dump(self.std_scaler, self.config['paths']['std_scaler'])
        joblib.dump(self.minmax_scaler, self.config['paths']['minmax_scaler_eval'])
        return df

    def compute_evaluation_metrics(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        # Placeholder for Information Coefficient (IC) and Rank IC (RIC)
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        ric = ic
        return {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'IC': ic, 'RIC': ric}

    def save_scaled_data(self, df):
        path = self.config['paths']['scaled_features']
        df.to_csv(path, index=False)
        return df

# ----------------------------
# Model Training Module
# ----------------------------
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None  # Replace with actual model instance (LSTM, Transformer, etc.)

    def train_model(self, X_train, y_train, X_val, y_val):
        # Placeholder training logic.
        # Insert your model training code and hyperparameter tuning (e.g., using Optuna) here.
        self.model = "trained_model_placeholder"
        joblib.dump(self.model, self.config['paths']['model'])
        return self.model

# ----------------------------
# Backtesting Module
# ----------------------------
class Backtester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, predictions, df):
        # Dummy backtesting logic: generate buy/sell signals based on predictions
        signals = ['Buy' if pred > 0 else 'Sell' for pred in predictions]
        df['Signal'] = signals
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Returns'] * (df['Signal'] == 'Buy').astype(int)
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        return df[['Signal', 'Cumulative_Returns']]

# ----------------------------
# Model Inference Module
# ----------------------------
class ModelInference:
    def __init__(self, config):
        self.config = config
        self.model = joblib.load(self.config['paths']['model'])
    
    def predict(self, X):
        # Replace with your actual prediction logic
        predictions = np.zeros(len(X))
        return predictions

# ----------------------------
# Pipeline Orchestrator
# ----------------------------
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.feature_refiner = FeatureRefiner(config)
        self.dynamic_selector = DynamicFeatureSelector(config)
        self.evaluator_scaler = EvaluatorScaler(config)
        self.trainer = ModelTrainer(config)
        self.backtester = Backtester(config)
        self.inference_module = ModelInference(config)

    def run_training_pipeline(self):
        # Data Cleaning & Preprocessing
        raw_df = self.preprocessor.load_raw_data()
        cleaned_df = self.preprocessor.clean_data(raw_df)
        cleaned_df = self.preprocessor.normalize_numeric(cleaned_df, fit=True)
        self.preprocessor.save_cleaned_data(cleaned_df)

        # Feature Engineering
        engineered_df = self.feature_engineer.compute_technical_indicators(cleaned_df)
        engineered_df = self.feature_engineer.compute_alpha158_features(engineered_df)
        engineered_df = self.feature_engineer.compute_candlestick_features(engineered_df)
        engineered_df = self.feature_engineer.compute_lag_features(engineered_df)
        engineered_df = self.feature_engineer.compute_additional_features(engineered_df)
        engineered_df = self.feature_engineer.denoise_close(engineered_df)
        engineered_df = self.feature_engineer.create_target_columns(engineered_df)
        self.feature_engineer.save_engineered_features(engineered_df)

        # Feature Refinement
        refined_df = self.feature_refiner.drop_highly_correlated(engineered_df)
        refined_df = self.feature_refiner.drop_low_mi_features(refined_df, 'future_avg_return')
        refined_df = self.feature_refiner.transform_target(refined_df)
        self.feature_refiner.save_refined_features(refined_df)

        # Dynamic Feature Selection
        dynamic_df = self.dynamic_selector.rolling_shap_analysis(refined_df)
        self.dynamic_selector.save_dynamic_features(dynamic_df)

        # Enhanced Evaluation & Scaling
        scaled_df = self.evaluator_scaler.dynamic_scaling(dynamic_df)
        self.evaluator_scaler.save_scaled_data(scaled_df)

        # Data Splitting (80/20)
        split_idx = int(0.8 * len(scaled_df))
        train_df = scaled_df.iloc[:split_idx].reset_index(drop=True)
        val_df = scaled_df.iloc[split_idx:].reset_index(drop=True)
        if 'future_avg_return' in train_df.columns:
            X_train = train_df.drop(columns=['future_avg_return'])
            y_train = train_df['future_avg_return']
            X_val = val_df.drop(columns=['future_avg_return'])
            y_val = val_df['future_avg_return']
        else:
            raise ValueError("Target column 'future_avg_return' not found.")
        
        # Model Training & Optimization
        self.trainer.train_model(X_train, y_train, X_val, y_val)
        print("Training pipeline completed and model saved.")

    def run_inference_pipeline(self):
        # Preprocess new (or live) data
        raw_df = self.preprocessor.load_raw_data()  # Update this for live data source as needed
        cleaned_df = self.preprocessor.clean_data(raw_df)
        cleaned_df = self.preprocessor.normalize_numeric(cleaned_df, fit=False)
        
        # Feature Engineering for Inference
        engineered_df = self.feature_engineer.compute_technical_indicators(cleaned_df)
        engineered_df = self.feature_engineer.compute_alpha158_features(engineered_df)
        engineered_df = self.feature_engineer.compute_candlestick_features(engineered_df)
        engineered_df = self.feature_engineer.compute_lag_features(engineered_df)
        engineered_df = self.feature_engineer.compute_additional_features(engineered_df)
        engineered_df = self.feature_engineer.denoise_close(engineered_df)
        engineered_df = self.feature_engineer.create_target_columns(engineered_df)
        
        # Apply Feature Refinement
        refined_df = self.feature_refiner.drop_highly_correlated(engineered_df)
        refined_df = self.feature_refiner.drop_low_mi_features(refined_df, 'future_avg_return')
        refined_df = self.feature_refiner.transform_target(refined_df)
        
        # Dynamic Feature Selection
        dynamic_df = self.dynamic_selector.rolling_shap_analysis(refined_df)
        
        # Enhanced Scaling
        scaled_df = self.evaluator_scaler.dynamic_scaling(dynamic_df)

        # Model Inference
        predictions = self.inference_module.predict(scaled_df)
        print("Inference completed.")

        # Backtesting: simulate trading strategy
        backtest_results = self.backtester.run_backtest(predictions, cleaned_df)
        print("Backtesting completed.")

        return predictions, backtest_results

if __name__ == "__main__":
    # Load configuration settings from YAML
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    pipeline = DataPipeline(config)
    
    # Uncomment the desired pipeline run:
    # For training:
    pipeline.run_training_pipeline()
    
    # For inference:
    # predictions, backtest_results = pipeline.run_inference_pipeline()
    # print("Predictions:", predictions)
    # print("Backtest Results:\n", backtest_results)
