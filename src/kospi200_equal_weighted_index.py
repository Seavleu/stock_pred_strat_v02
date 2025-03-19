'''
We assigned an equal weight to all stock to prevent bias training.
This script will compute the daily returns for each stock, then creates
future targets (next day, day after next, and 5-day forecast) 

Objective:
- Equal weighted index: is computed by averaging the daily returns across all stocks
for each DATE and then converting these returns into a cumulative index start at 100
- Future targets: next_day_index, day_after_next_index, future_5day_index
'''
import os
import pandas as pd
import numpy as np

def compute_daily_returns(df):
    # sort by ticker & date, then compute daily return for each stock
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    return df

def compute_equal_weighted_index(df):
    # group by date and avg the daily returns across all stocks  
    daily_avg = df.groupby("date")["daily_return"].mean().reset_index()
    daily_avg["daily_return"] = daily_avg["daily_return"].fillna(0)
    # cumulative index with a base value of 100
    daily_avg["cumulative_index"] = 100 * np.cumprod(1 + daily_avg["daily_return"])
    return daily_avg

def assign_future_targets(index_df):
    """
    Create future target columns:
      - next_day_index: the index for the next day.
      - day_after_next_index: the index for the day after next.
      - future_5day_index: the index for 5 days ahead.
    """
    index_df = index_df.sort_values("date").reset_index(drop=True)
    index_df["next_day_index"] = index_df["cumulative_index"].shift(-1)
    index_df["day_after_next_index"] = index_df["cumulative_index"].shift(-2)
    index_df["future_5day_index"] = index_df["cumulative_index"].shift(-5)
    # drop if NaN
    index_df.dropna(inplace=True)
    return index_df

def main():
    input_file = "data/mapping/korean_stock_data_with_ticker.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} does not exist.")
    df = pd.read_csv(input_file, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {input_file}.")
    
    # filter out any rows without tickers -> unknown companies
    df = df[df["ticker"].notnull()].copy()
    
    # daily returns per stock
    df = compute_daily_returns(df)
    
    # Compute the equal-weighted index from the daily returns
    index_df = compute_equal_weighted_index(df)
    print("Equal-weighted index (first 5 rows):")
    print(index_df.head())
    
    # Assign future targets (next day, day after next, 5-day ahead)
    index_df = assign_future_targets(index_df)
    print("Index with future targets (first 5 rows):")
    print(index_df.head())
    
    # Save the intermediate index file (with future targets) in interim folder
    interim_path = "data/interim/kospi_200_equal_weighted_index_with_targets.csv"
    os.makedirs(os.path.dirname(interim_path), exist_ok=True)
    index_df.to_csv(interim_path, index=False)
    print(f"Saved intermediate index with targets to {interim_path}")
    
    final_path = "data/processed/kospi_200_equal_weighted_index_final.csv"
    index_df.to_csv(final_path, index=False)
    print(f"Saved final processed index to {final_path}")

if __name__ == "__main__":
    main()
