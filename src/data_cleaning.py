import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def clean_and_preprocess(csv_file, output_dir="data/processed"):
    """
    Clean missing values, handle anomalies, normalize, and preprocess raw stock data.
    Saves cleaned data to a CSV in the specified output directory.
    
    Parameters:
        csv_file (str): Path to the raw CSV file containing columns:
                        [opening_price, highest_price, lowest_price, closing_price,
                         trading_volume, num_of_shares, company, timestamp]
        output_dir (str): Directory where the cleaned CSV file will be saved.
    
    Returns:
        DataFrame: The cleaned and preprocessed DataFrame.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    
    df.columns = df.columns.str.strip()
    
    # Fill missing values using forward-fill and backward-fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Handle anomalies/outliers using IQR clipping on numeric columns.
    # Here, we process the key numeric columns.
    numeric_cols = ["opening_price", "highest_price", "lowest_price", "closing_price", "trading_volume", "num_of_shares"]
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Normalize numeric data using MinMaxScaler.
    # Note: You might choose not to scale 'num_of_shares' if it's not a predictive feature.
    scale_cols = ["opening_price", "highest_price", "lowest_price", "closing_price", "trading_volume"]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    # Drop any remaining NaN rows.
    df.dropna(inplace=True)
    
    # Save cleaned data.
    base_filename = os.path.basename(csv_file).replace("_stock_data.csv", "_cleaned_data.csv")
    output_path = os.path.join(output_dir, base_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    
    return df

'''Usage Example:
if __name__ == "__main__":
    input_csv = "data/raw/korean_stock_data.csv"
    cleaned_df = clean_and_preprocess(input_csv)
'''
