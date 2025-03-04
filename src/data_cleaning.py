# Functions for handling missing values, outliers, or merges.
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def clean_and_preprocess(csv_file, output_dir="data/processed"):
    """
    Clean missing values, handle anomalies, normalize, and preprocess data.
    Saves cleaned data to a CSV in the specified output directory.
    
    Parameters:
        csv_file (str): Path to the raw CSV file containing columns:
                        ['Date','Open','High','Low','Close','Volume']
        output_dir (str): Directory where the cleaned CSV file will be saved.
    
    Returns:
        DataFrame: The cleaned and preprocessed DataFrame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read raw data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    
    # Rename columns if necessary (e.g., matching your naming convention)
    # Example: If your CSV has columns like 'opening_price', 'closing_price', etc.
    rename_cols = {
        "Date": "timestamp",
        "Open": "opening_price",
        "High": "highest_price",
        "Low": "lowest_price",
        "Close": "closing_price",
        "Volume": "trading_volume"
    }
    
    for old_col, new_col in rename_cols.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    
    # Fill missing values: forward fill, then backward fill 
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Handle anomalies/outliers using IQR clipping on numeric columns
    numeric_cols = ["opening_price", "highest_price", "lowest_price", "closing_price", "trading_volume"]
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Normalize numeric data using MinMaxScaler
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Drop any remaining NaN rows (if any edge cases remain)
    df.dropna(inplace=True)
    
    # Save cleaned data
    # Example CSV name: "AAPL_cleaned_stock_data.csv"
    base_filename = os.path.basename(csv_file).replace("_raw_data.csv", "_cleaned_data.csv")
    output_path = os.path.join(output_dir, base_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    
    return df

'''Usage Example:
if __name__ == "__main__":
    # Suppose you have already downloaded AAPL_raw_data.csv
    input_csv = "data/raw/AAPL_raw_data.csv"
    cleaned_df = clean_and_preprocess(input_csv)
'''