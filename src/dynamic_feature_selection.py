'''
Dynamic Feature Selection using Rolling-Window SHAP Analysis. 

We implements a dynamic feature selection strategy here, first it
loads the refined dataset from our 'alpha158_enhanced_features_scaled.csv', then uses
a rolling-window to compute SHAP values with an XGBoost model,
-> then aggregates the feature importance over time, and selects features
that are consistently predictive under different market conditions

Detect shorter-term market changes, is particularly useful in high-volatility periods
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use("Agg")

def dynamic_feature_selection(df, window_size=200, step_size=50):
    """ 
    first this func() will slides a window over refined dataset and fits an XGBoost model
    in each window, compute SHAP values, and aggregate the absolute SHAP values for each feature.
    this will help us to dynamically asses which features are consistently important under different
    market conditions.
    
    Args:
        df (DataFrame): Refined dataset containing 'timestamp', 'closing_price',
                        and feature columns.
        window_size (int): Number of rows to use in each rolling window.
        step_size (int): Step size to slide the window.
    
    Returns:
        aggregated_shap (Series): Average absolute SHAP value for each feature.
    """
    # prepare features (X) and target (y)
    # feature_cols = [col for col in df.columns if col not in ["timestamp", "closing_price"]]
    
    # exclude non-feature columns, including lag features and market-wide indicators
    excluded_cols = ["timestamp", "closing_price", "company"] + [col for col in df.columns if "lag" in col or "KOSPI" in col or "FX_rate" in col]
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols]
    y = df["closing_price"]

    # initialize accumulator for SHAP values
    shap_accumulator = {feature: [] for feature in feature_cols}
    n_windows = 0

    # loop over rolling windows
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        X_window = X.iloc[start:end]
        y_window = y.iloc[start:end]
        ''
        X_window.fillna(0, inplace=True)  # replace NaNs with 0
        y_window.fillna(method="ffill", inplace=True)  # Forward-fill missing target vals
        
        # fit an XGBoost model on the window
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_window, y_window)
        
        # create SHAP explainer and compute SHAP vals on the window
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_window)
        
        # aggregate the absolute SHAP values per feature
        abs_shap = np.abs(shap_values)
        # sum across samples -> rows
        window_shap_importance = np.mean(abs_shap, axis=0)
        
        # append the results -> each feature
        for i, feature in enumerate(feature_cols):
            shap_accumulator[feature].append(window_shap_importance[i])
        
        n_windows += 1
        print(f"Processed window {n_windows}: rows {start} to {end}")

    # compute the average SHAP importance per feature over all windows
    aggregated_shap = {feature: np.mean(values) for feature, values in shap_accumulator.items()}
    aggregated_shap_series = pd.Series(aggregated_shap).sort_values(ascending=False)
    
    return aggregated_shap_series

def plot_aggregated_shap(aggregated_shap, output_path="docs/dynamic_shap_importance.png"):
    """Plot aggregated SHAP importance as a bar chart and save to file."""
    plt.figure(figsize=(10,6))
    aggregated_shap.plot(kind="bar")
    plt.title("Aggregated Dynamic SHAP Feature Importances")
    plt.ylabel("Average |SHAP| Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Dynamic SHAP importance plot saved to '{output_path}'.")

def main():
    # loaded data -> output from feature_refinement.py
    refined_csv = "data/processed/alpha158_enhanced_features_scaled.csv"
    if not os.path.exists(refined_csv) or os.path.getsize(refined_csv) == 0:
        sys.exit(f"Error: {refined_csv} is missing or empty. Please run the feature refinement pipeline first.")
    
    df = pd.read_csv(refined_csv, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {refined_csv}.")
    
    # dynamic feature selection using rolling windows
    aggregated_shap = dynamic_feature_selection(df, window_size=200, step_size=15) # test: 50, 25,
    print("Aggregated Dynamic SHAP Importances:")
    print(aggregated_shap)
    
    plot_aggregated_shap(aggregated_shap)
    
    # threshold for selecting features based on aggregated SHAP vals
    shap_cutoff = np.percentile(aggregated_shap.values, 50)  # top 50%
    selected_features = aggregated_shap[aggregated_shap >= shap_cutoff].index.tolist()
    
    print(f"Selected dynamic features (above cutoff {shap_cutoff:.4f}): {selected_features}")
    
    keep_cols = ["timestamp", "closing_price"] + selected_features
    dynamic_selected_df = df[keep_cols].copy()
    output_csv = "data/processed/dynamic_alpha_selected_features.csv"
    dynamic_selected_df.to_csv(output_csv, index=False)
    print(f"Dynamic selected dataset saved to {output_csv}")

if __name__ == "__main__":
    main()
