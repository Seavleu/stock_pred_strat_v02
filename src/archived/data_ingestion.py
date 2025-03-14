# Scripts/functions to load data (possibly in chunks) and handle database or API connections
import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(tickers, start_date, end_date, output_dir="data/raw"):
    """
    Fetch historical stock data for the provided tickers from Yahoo Finance.
    Saves each ticker's raw data as a CSV in the specified output directory.
    
    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        output_dir (str): Directory where raw CSV files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download data for all tickers in one call for efficiency
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", group_by='ticker', auto_adjust=False)
    
    # Handle multiple tickers vs single ticker
    if isinstance(tickers, list) and len(tickers) > 1:
        # Separate data for each ticker
        for ticker in tickers:
            df_ticker = data[ticker].copy()
            df_ticker.reset_index(inplace=True)
            # Save raw CSV for each ticker
            csv_filename = os.path.join(output_dir, f"{ticker}_raw_data.csv")
            df_ticker.to_csv(csv_filename, index=False)
            print(f"Saved raw data for {ticker} to {csv_filename}")
    else:
        # Single ticker scenario
        ticker = tickers[0] if isinstance(tickers, list) else tickers
        data.reset_index(inplace=True)
        csv_filename = os.path.join(output_dir, f"{ticker}_raw_data.csv")
        data.to_csv(csv_filename, index=False)
        print(f"Saved raw data for {ticker} to {csv_filename}")

    """Usage Example:
    if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    fetch_stock_data(tickers, start_date, end_date)
    """