# Orchestrates data ingestion, cleaning, and feature engineering
#!/usr/bin/env bash

# Example usage:
# bash scripts/run_data_pipeline.sh

echo "=== Fetching Stock Data ==="
python -c "
from src.data_ingestion import fetch_stock_data
tickers = ['AAPL', 'GOOGL']
fetch_stock_data(tickers, '2020-01-01', '2023-01-01')
"

echo "=== Cleaning & Preprocessing Data ==="
python -c "
import glob
from src.data_cleaning import clean_and_preprocess

raw_files = glob.glob('data/raw/*_raw_data.csv')
for f in raw_files:
    clean_and_preprocess(f)
"

echo "=== Data Pipeline Completed ==="
