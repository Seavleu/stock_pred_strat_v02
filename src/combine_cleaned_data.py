import pandas as pd
import glob
import os

def combine_cleaned_files(input_dir="data/processed", output_file="data/processed/combined_cleaned_stock_data.csv"):
    # find all cleaned CSV files that follow the pattern *_cleaned_data.csv
    all_files = glob.glob(os.path.join(input_dir, "*_cleaned_data.csv"))
    if not all_files:
        print("No cleaned data files found. Please run the cleaning pipeline first.")
        return
    
    df_list = [pd.read_csv(file, parse_dates=['timestamp']) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.sort_values("timestamp", inplace=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined cleaned data saved to {output_file}")

if __name__ == "__main__":
    combine_cleaned_files()
