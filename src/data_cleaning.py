# src/data_cleaning.py

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.preprocessing import MinMaxScaler

class DataCleaner:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_raw_data(self):
        """Load raw Korean stock data and standardize column names."""
        df = pd.read_csv(self.config['data_paths']['raw_data'])
        # Trim and lowercase all column names
        df.columns = [col.strip().lower() for col in df.columns]
        # Rename cols using mapping from config
        rename_dict = self.config.get('rename_columns', {})
        if rename_dict:
            df = df.rename(columns=rename_dict)
        return df

    def clean_company_names(self, df):
        """Clean company names by removing suffixes and unwanted characters."""
        if 'company' in df.columns:
            df['company'] = df['company'].astype(str).str.strip()
            # Convert company names to lowercase
            df['company'] = df['company'].str.lower()
            # Remove common suffixes like 'Co., Ltd.' (case-insensitive)
            df['company'] = df['company'].apply(lambda x: re.sub(r'\s*(co\.?,?\s*ltd\.?)$', '', x, flags=re.IGNORECASE))
            # Remove any non-alphanumeric characters if needed
            df['company'] = df['company'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        return df

    def filter_invalid_companies(self, df):
        """Filter out rows with missing or 'unknown' company names."""
        if 'company' in df.columns:
            df = df[df['company'].notna()]
            df = df[~df['company'].str.lower().str.contains("unknown")]
        return df

    def group_duplicates(self, df):
        """Group by company and date to average duplicate records."""
        if 'date' in df.columns and 'company' in df.columns:
            df = df.groupby(['date', 'company'], as_index=False).mean()
        return df

    def quality_filtering(self, df):
        """Drop stocks with too many missing values or low liquidity."""
        # Maximum allowed missing percentage per row (e.g., 20%)
        missing_thresh = self.config['quality_filters'].get('max_missing_pct', 0.2)
        # Minimum average volume threshold for liquidity
        volume_thresh = self.config['quality_filters'].get('min_avg_volume', 10000)
        
        # Drop rows that do not have sufficient non-missing columns
        df = df.dropna(thresh=int((1 - missing_thresh) * len(df.columns)))
        
        # Filter out stocks with low liquidity based on 'volume'
        if 'volume' in df.columns:
            df = df[df['volume'] >= volume_thresh]
        return df

    def iqr_clip(self, df):
        """Apply IQR-based clipping to remove outliers in numeric columns."""
        numeric_cols = self.config['scaling']['numeric_columns']
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def global_normalization(self, df):
        """Normalize numeric cols globally using MinMaxScaler."""
        numeric_cols = self.config['scaling']['numeric_columns']
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        joblib.dump(self.scaler, self.config['paths']['global_scaler'])
        return df

    def assign_ticker(self, df):
        """Integrate ticker mapping into the cleaned data while preserving ticker format as string."""
        ticker_path = self.config['data_paths'].get('ticker_data')
        if ticker_path:
            # Read ticker mapping with ticker as string to preserve leading zeros
            ticker_df = pd.read_csv(ticker_path, dtype={'ticker': str})
            # Standardize ticker mapping column names
            ticker_df.columns = [col.strip().lower() for col in ticker_df.columns]
            # Merge assuming ticker mapping has cols 'company' and 'ticker'
            df = df.merge(ticker_df[['company', 'ticker']], on='company', how='left')
        return df

    def save_cleaned_data(self, df):
        """Save the overall cleaned dataset and individual company files."""
        cleaned_path = self.config['paths']['cleaned_data']
        df.to_csv(cleaned_path, index=False)
        
        extracted_dir = self.config['paths']['extracted_dir']
        import os
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Save individual files: use ticker if available, else use company name
        if 'ticker' in df.columns:
            for ticker, group in df.groupby('ticker'):
                group.to_csv(f"{extracted_dir}/{ticker}.csv", index=False)
        else:
            for company, group in df.groupby('company'):
                safe_name = re.sub(r'\W+', '_', company)
                group.to_csv(f"{extracted_dir}/{safe_name}.csv", index=False)
        return df

    def run_cleaning_pipeline(self):
        """Run the complete data cleaning pipeline."""
        df = self.load_raw_data()
        df = self.clean_company_names(df)
        df = self.filter_invalid_companies(df)
        df = self.group_duplicates(df)
        df = self.quality_filtering(df)
        df = self.iqr_clip(df)
        df = self.global_normalization(df)
        df = self.assign_ticker(df)
        df = self.save_cleaned_data(df)
        return df

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    cleaner = DataCleaner(config)
    cleaned_df = cleaner.run_cleaning_pipeline()
    print("Data cleaning completed. Cleaned data saved to:", config['paths']['cleaned_data'])
