import os
import pandas as pd
import numpy as np
import re

def clean_company_name(company):
    """
    Clean the company name by converting to lowercase,
    removing punctuation and extra spaces.
    """
    s = str(company).strip().lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def assign_tickers(df, ticker_file="data/raw/ticker.csv"):
    """
    Assign tickers to the processed stock data based on a ticker mapping file.
    The mapping file is expected to have columns: 'ticker' and 'company'.
    Returns the dataframe with a new 'ticker' column.
    """
    df_ticker = pd.read_csv(ticker_file)
    df_ticker["company_clean"] = df_ticker["company"].apply(clean_company_name)
    mapping = dict(zip(df_ticker["company_clean"], df_ticker["ticker"]))
    
    # Clean company names in df
    if "company" in df.columns:
        df["company_clean"] = df["company"].apply(clean_company_name)
        df["ticker"] = df["company_clean"].apply(lambda x: mapping.get(x, None))
        df.drop(columns=["company_clean"], inplace=True)
    else:
        df["ticker"] = np.nan
    return df

def filter_top_200_by_market_cap(df):
    """
    Compute market cap as close * share (using unscaled values) and
    filter the dataset to include only the top 200 companies by average market cap.
    """
    # if market_cap is not already present, compute it
    if "market_cap" not in df.columns:
        # Ensure that 'close' and 'share' columns are numeric
        df["market_cap"] = df["close"].astype(float) * df["share"].astype(float)
    
    # Group by company and compute the average market cap
    cap_df = df.groupby("company")["market_cap"].mean().reset_index()
    top_200 = cap_df.sort_values("market_cap", ascending=False).head(200)
    filtered_df = df[df["company"].isin(top_200["company"])]
    return filtered_df

def save_individual_files(df, output_dir="data/processed/individual_stocks"):
    """
    Save individual stock files from the dataframe.
    The filename is the ticker if available; if ticker is missing, use the company name.
    """
    os.makedirs(output_dir, exist_ok=True)
    for _, group in df.groupby("company"):
        key = group["ticker"].iloc[0] if pd.notnull(group["ticker"].iloc[0]) else group["company"].iloc[0]
        # Clean key for filename
        filename = f"{str(key).strip().lower().replace(' ', '_')}.csv"
        output_path = os.path.join(output_dir, filename)
        group = group.sort_values("date")
        group.to_csv(output_path, index=False)
        print(f"Saved file: {output_path}")

def main():
    # Load processed cleaned dataset (assumed to have columns: date, company, open, high, low, close, volume, share)
    input_csv = "data/processed/korean_stock_data_cleaned.csv"
    if not os.path.exists(input_csv) or os.path.getsize(input_csv) == 0:
        sys.exit(f"Error: {input_csv} is missing or empty.")
    
    df = pd.read_csv(input_csv, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {input_csv}.")

    # Assign tickers based on ticker.csv
    df = assign_tickers(df, ticker_file="data/raw/ticker.csv")
    
    # Compute market cap and filter top 200 companies
    df_top200 = filter_top_200_by_market_cap(df)
    print(f"Filtered dataset to top 200 companies by market cap; remaining rows: {len(df_top200)}")
    
    # Save full cleaned dataset with tickers
    final_output = "data/processed/korean_stock_data_final.csv"
    df_top200.to_csv(final_output, index=False)
    print(f"Saved full cleaned dataset with tickers to {final_output}")
    
    # Create ticker-only dataset (drop company column)
    if "company" in df_top200.columns:
        df_ticker_only = df_top200.drop(columns=["company"])
    else:
        df_ticker_only = df_top200.copy()
    ticker_only_output = "data/processed/korean_stock_data_ticker_only.csv"
    df_ticker_only.to_csv(ticker_only_output, index=False)
    print(f"Saved ticker-only dataset to {ticker_only_output}")
    
    # Save individual stock files
    save_individual_files(df_top200, output_dir="data/processed/individual_stocks")

if __name__ == "__main__":
    main()
