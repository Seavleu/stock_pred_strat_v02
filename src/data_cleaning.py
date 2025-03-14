import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

# def clean_company_name(company):
#     """
#     cleans the company name by removing common suffixes, punctuation, and unnecessary characters.
#     """
#     s = str(company).strip().lower()

#     patterns_to_remove = [
#         r'\s*\.co\.,?\s*ltd\.?$',
#         r'\s*co\.,?\s*ltd\.?$',
#         r'\s*\.$',
#         r'^\s*"',
#         r'"\s*$',
#         r'\s*,?inc\.?$',
#         r'\s*company limited$',
#         r'\s*corporation$',
#         r'\s*corp$',
#         r'\s*crop$',
#         r'\s*ltd$',
#         r'\s*company$',
#         r'\s*\.?company$',
#         r'\s*,?company$',
#         r'\s*,'
#     ]
#     for pattern in patterns_to_remove:
#         s = re.sub(pattern, '', s, flags=re.IGNORECASE).strip()
#     return s

def filter_low_quality_stocks(df):
    """
    applies multiple quality checks to filter out unreliable stock data.
    """

    # drop companies with too many missing values (>30% missing price/volume data)
    df = df.dropna(thresh=int(df.shape[1] * 0.7))

    # drop stocks with low liquidity (less than 10,000 avg. daily volume)
    low_volume_companies = df.groupby("company")["volume"].mean()
    low_volume_companies = low_volume_companies[low_volume_companies < 10000].index
    df = df[~df["company"].isin(low_volume_companies)]

    # drop stocks with extreme price volatility (std dev > mean * 3)
    high_volatility_companies = df.groupby("company")["close"].std() > df.groupby("company")["close"].mean() * 3
    high_volatility_companies = high_volatility_companies[high_volatility_companies].index
    df = df[~df["company"].isin(high_volatility_companies)]

    # drop stocks with very low price movement (stagnant stocks)
    price_range = df.groupby("company")["close"].max() - df.groupby("company")["close"].min()
    stagnant_companies = price_range[price_range < df.groupby("company")["close"].mean() * 0.05].index
    df = df[~df["company"].isin(stagnant_companies)]

    # drop companies with frequent zero values (>30% of prices or volume are zero)
    zero_counts = (df[['open', 'high', 'low', 'close', 'volume']] == 0).sum()
    frequent_zero_companies = zero_counts[zero_counts > df.shape[0] * 0.3].index
    df = df[~df["company"].isin(frequent_zero_companies)]

    return df

def clean_and_preprocess(csv_file, 
                         output_csv="data/processed/korean_stock_data_cleaned.csv",
                         output_dir_extracted="data/processed/korean_stock_extracted",
                         missing_threshold=0.3):
    """
    cleans missing values, removes duplicates, handles anomalies, and normalizes stock data.
    applies additional quality filters for high-reliability stock data.
    """

    # create directories if not exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(output_dir_extracted, exist_ok=True)

    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df.columns = df.columns.str.strip()

    # rename columns to match standard format
    column_mapping = {
        "timestamp": "date",
        "opening_price": "open",
        "highest_price": "high",
        "lowest_price": "low",
        "closing_price": "close",
        "trading_volume": "volume",
        "num_of_shares": "share",
        "company": "company"
    }
    df.rename(columns=column_mapping, inplace=True)

    # clean company names
    df["company"] = df["company"].astype(str) 
    df = df[df["company"].notnull() & df["company"].str.strip().ne("")]
    df = df[~df["company"].str.lower().eq("unknown")]

    # remove 88 "unknown" companies (unknown1-unknown88)
    df = df[~df["company"].str.match(r"^unknown([1-8][0-9]?|88)$", na=False)]

    # group by company and date, averaging duplicate records
    df = df.groupby(["company", "date"], as_index=False).mean()

    # fill missing values with forward-fill and backward-fill
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # apply anomaly filtering (iqr clipping for outliers)
    numeric_cols = ["open", "high", "low", "close", "volume", "share"]
    for col in numeric_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # apply stock quality filters
    df = filter_low_quality_stocks(df)

    # normalize numeric columns
    scale_cols = ["open", "high", "low", "close", "volume"]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df.dropna(inplace=True)

    # standardize date format and sort
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "company"], inplace=True)

    df.drop_duplicates(subset=["date", "company"], inplace=True)

    # ensure no negative or zero prices
    df = df[df["close"] > 0]

    df.reset_index(drop=True, inplace=True)

    # reorder columns (date first)
    cols = ["date"] + [col for col in df.columns if col != "date"]
    df = df[cols]

    # normalize stock data per company
    def normalize_group(group):
        group = group.sort_values("date").reset_index(drop=True)
        scaler_mm = MinMaxScaler(feature_range=(-1, 1))
        group["close"] = scaler_mm.fit_transform(group[["close"]])
        other_cols = ["open", "high", "low", "volume"]
        scaler_std = StandardScaler()
        group[other_cols] = scaler_std.fit_transform(group[other_cols])
        return group

    df_norm = df.groupby("company").apply(normalize_group).reset_index(drop=True)

    # save overall cleaned data
    df_norm.to_csv(output_csv, index=False)
    print(f"✅ saved cleaned data to {output_csv}")

    # save cleaned data by company
    for company, group in df_norm.groupby("company"):
        company_output_path = os.path.join(output_dir_extracted, f"{company.replace(' ', '_')}_cleaned.csv")
        group.to_csv(company_output_path, index=False)

    print(f"✅ processed {df['company'].nunique()} companies.")

    return df_norm

def save_combined_clean_data(df, output_path="data/processed/korean_stock_combined_clean_data.csv"):
    """
    saves the combined cleaned data to a specified output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ saved combined cleaned data to {output_path}")

if __name__ == "__main__":
    input_csv = "data/raw/korean_stock_data.csv"
    cleaned_df = clean_and_preprocess(input_csv)
    save_combined_clean_data(cleaned_df)
    print(f"✅ final dataset contains {cleaned_df['company'].nunique()} high-quality companies.")
