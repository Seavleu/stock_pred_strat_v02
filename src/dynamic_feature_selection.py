"""
dynamic feature selection using rolling-window shap analysis.

this module loads the refined dataset (with 'date' and 'future_avg_return'),
then slides a window over the data, fits an xgboost model in each window,
computes shap values, aggregates the absolute shap values for each feature,
and selects features that are consistently predictive under different market conditions.

this approach helps detect shorter-term market changes, which is especially useful in high-volatility periods.
"""

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
    slides a window over the refined dataset and fits an xgboost model in each window.
    computes shap values and aggregates the absolute shap values for each feature.
    
    args:
        df (dataframe): refined dataset containing 'date', 'future_avg_return',
                        and feature columns.
        window_size (int): number of rows to use in each rolling window.
        step_size (int): step size to slide the window.
    
    returns:
        aggregated_shap (series): average absolute shap value for each feature.
    """
    # prepare features (X) and target (y); exclude 'date' and target column
    feature_cols = [col for col in df.columns if col not in ["date", "future_avg_return"]]
    X = df[feature_cols]
    y = df["future_avg_return"]

    # initialize accumulator for shap values
    shap_accumulator = {feature: [] for feature in feature_cols}
    n_windows = 0

    # loop over rolling windows
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        X_window = X.iloc[start:end]
        y_window = y.iloc[start:end]
        
        # fit an xgboost model on the window
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_window, y_window)
        
        # create shap explainer using interventional perturbation
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        # pass check_additivity=False in the shap_values call
        shap_values = explainer.shap_values(X_window, check_additivity=False)
        
        # aggregate the absolute shap values per feature
        abs_shap = np.abs(shap_values)
        window_shap_importance = np.mean(abs_shap, axis=0)
        
        # append the results for each feature
        for i, feature in enumerate(feature_cols):
            shap_accumulator[feature].append(window_shap_importance[i])
        
        n_windows += 1
        print(f"processed window {n_windows}: rows {start} to {end}")

    # compute average shap importance per feature over all windows
    aggregated_shap = {feature: np.mean(values) for feature, values in shap_accumulator.items()}
    aggregated_shap_series = pd.Series(aggregated_shap).sort_values(ascending=False)
    
    return aggregated_shap_series

def plot_aggregated_shap(aggregated_shap, output_path="docs/dynamic_shap_importance.png"):
    """
    plot aggregated shap importance as a bar chart and save to file.
    """
    plt.figure(figsize=(10,6))
    aggregated_shap.plot(kind="bar")
    plt.title("Aggregated Dynamic SHAP Feature Importances")
    plt.ylabel("Average |SHAP| Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"dynamic shap importance plot saved to '{output_path}'.")

def main():
    # load refined dataset (output from feature refinement pipeline)
    refined_csv = "data/processed/refined_features.csv"
    if not os.path.exists(refined_csv) or os.path.getsize(refined_csv) == 0:
        sys.exit(f"error: {refined_csv} is missing or empty. please run the feature refinement pipeline first.")
    
    df = pd.read_csv(refined_csv, parse_dates=["date"])
    print(f"loaded {len(df)} rows from {refined_csv}.")
    
    # dynamic feature selection using rolling windows
    aggregated_shap = dynamic_feature_selection(df, window_size=200, step_size=50)
    print("aggregated dynamic shap importances:")
    print(aggregated_shap)
    
    plot_aggregated_shap(aggregated_shap)
    
    # set threshold for selecting features based on aggregated shap values
    shap_cutoff = np.percentile(aggregated_shap.values, 50)  # top 50%
    selected_features = aggregated_shap[aggregated_shap >= shap_cutoff].index.tolist()
    
    print(f"selected dynamic features (above cutoff {shap_cutoff:.4f}): {selected_features}")
    
    # create final dataset with selected features; include date and target future_avg_return
    keep_cols = ["date", "future_avg_return"] + selected_features
    dynamic_selected_df = df[keep_cols].copy()
    output_csv = "data/processed/dynamic_selected_features.csv"
    dynamic_selected_df.to_csv(output_csv, index=False)
    print(f"dynamic selected dataset saved to {output_csv}")

if __name__ == "__main__":
    main()
