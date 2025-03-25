"""
dynamic feature selection using rolling-window shap analysis.

this module loads the refined dataset (with 'date' and 'future_avg_return'),
then slides a window over the data, fits an xgboost model in each window,
computes shap values, aggregates the absolute shap values for each feature,
and selects features that are consistently predictive under different market conditions.

this approach helps detect shorter-term market changes, which is especially useful in high-volatility periods.

**side note & benefit**
- traditional models might use static features, assuming the same set of factors always influence stock price movement,
however, market dynamic shift over time, meaning different indicators (volatility, momentum, volume) may gain or lose imortances
└──> solution: rolling-window feature selection, will continuously evaluates which features matter the most (adapt to market trends 
-> i need to create more macroeconomic feature for identify trends )
- For example, if a hedge fund wants to optimize its stock trading algorithm, they use technical indicators (RSI, MACD, Bolinger Bands) 
and fundamental data to predict. 

- Dynamic Features Selection :
    └──> The model will start to slides over recent data (e.g last 200d -> moving forward 50d/time) + each window will determine which indicators
    currently matter most for prediction target value (returns price)
    └──> If volatility and momentum indicators become more important due to market uncertainty, the model prioritize them
    └──> If fundamental indicators become more relevant -> shift focus accordingly
- Refine Trading Signal :
    └──> The fund now trades based on the most up-to-date info
    └──> Dynamically adapt to changing market conditions -> +profitability & reducing risk

"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg") 

def dynamic_feature_selection(df, window_size=200, step_size=50):
    """
    Slides a window over the refined dataset and computes the average absolute SHAP 
    values for each feature using an XGBoost model in each window.
    
    Args:
        df (DataFrame): Refined dataset containing 'date', 'future_avg_return', and feature columns.
        window_size (int): Number of rows per window.
        step_size (int): Number of rows to advance the window each step.
    
    Returns:
        Series: Aggregated (mean) absolute SHAP importance for each feature.
    """
    # Prepare features (X) and target (y)
    feature_cols = [col for col in df.columns if col not in ["date", "future_avg_return"]]
    X = df[feature_cols]
    y = df["future_avg_return"]

    # Initialize accumulator for SHAP values per feature
    shap_accumulator = {feature: [] for feature in feature_cols}
    n_windows = 0

    # Loop over rolling windows
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        X_window = X.iloc[start:end]
        y_window = y.iloc[start:end]
        
        # Fit XGBoost model on current window
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_window, y_window)
        
        # Create SHAP explainer using interventional perturbation
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_window, check_additivity=False)
        
        # Aggregate absolute SHAP values for current window
        abs_shap = np.abs(shap_values)
        window_shap_importance = np.mean(abs_shap, axis=0)
        
        for i, feature in enumerate(feature_cols):
            shap_accumulator[feature].append(window_shap_importance[i])
        
        n_windows += 1
        print(f"Processed window {n_windows}: rows {start} to {end}")
    
    # Compute the average SHAP importance per feature over all windows
    aggregated_shap = {feature: np.mean(values) for feature, values in shap_accumulator.items()}
    aggregated_shap_series = pd.Series(aggregated_shap).sort_values(ascending=False)
    
    return aggregated_shap_series

def plot_aggregated_shap(aggregated_shap, output_path="docs/dynamic_shap_importance.png"):
    """
    Plots the aggregated SHAP importances as a bar chart and saves the figure.
    """
    plt.figure(figsize=(10, 6))
    aggregated_shap.plot(kind="bar")
    plt.title("Aggregated Dynamic SHAP Feature Importances")
    plt.ylabel("Average |SHAP| Value")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Dynamic SHAP importance plot saved to '{output_path}'.")

def main():
    refined_csv = "data/processed/refined_features.csv"
    if not os.path.exists(refined_csv) or os.path.getsize(refined_csv) == 0:
        sys.exit(f"Error: {refined_csv} is missing or empty. Please run the feature refinement pipeline first.")
    
    df = pd.read_csv(refined_csv, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {refined_csv}.")
    
    # Run dynamic feature selection using rolling windows
    aggregated_shap = dynamic_feature_selection(df, window_size=200, step_size=50)
    print("Aggregated Dynamic SHAP Importances:")
    print(aggregated_shap)
    
    plot_aggregated_shap(aggregated_shap, output_path="docs/dynamic_shap_importance.png")
    
    # Select features whose aggregated SHAP values are above the 50th percentile
    shap_cutoff = np.percentile(aggregated_shap.values, 50) # 50 good starting point for balance, 70% better optimizing, 30% for market behavior 
    selected_features = aggregated_shap[aggregated_shap >= shap_cutoff].index.tolist()
    print(f"Selected dynamic features (above cutoff {shap_cutoff:.4f}): {selected_features}")
    
    # Create the final dataset with selected features
    keep_cols = ["date", "future_avg_return"] + selected_features
    dynamic_selected_df = df[keep_cols].copy()
    output_csv = "data/processed/dynamic_selected_features.csv"
    dynamic_selected_df.to_csv(output_csv, index=False)
    print(f"Dynamic selected dataset saved to {output_csv}")

if __name__ == "__main__":
    main()
