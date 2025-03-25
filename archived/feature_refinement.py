"""
feature refinement script

removes highly correlated features and drops low-mi (or low-shap) features
from your selected feature set. saves the refined dataset for re-training
and further model optimization.

Target calculation: add_future_return_targets() function creates three new cols 
for forecasting -> then use the average return as the final target.


"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

#############################################
# Target Generation
#############################################
def add_future_return_targets(df):
    """
    Compute target returns:
      - next_day_return = (close_{t+1} / close - 1)
      - day_after_next_return = (close_{t+2} / close - 1)
      - future_avg_return = average of the above two

    Rows without sufficient future data are dropped.
    """
    df = df.copy()
    epsilon = 1e-10  # small constant to avoid division by zero
    df['next_day_return'] = df['close'].shift(-1) / (df['close'] + epsilon) - 1
    df['day_after_next_return'] = df['close'].shift(-2) / (df['close'] + epsilon) - 1
    df['future_avg_return'] = (df['next_day_return'] + df['day_after_next_return']) / 2
    # Replace any infinite values and drop rows missing future targets
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['future_avg_return'], inplace=True)
    return df

#############################################
# Feature Correlation Filtering
#############################################
def remove_correlated_features(df, features, threshold=0.99):
    """
    Remove features that are highly correlated with one another.
    
    Args:
        df (DataFrame): The dataset containing features.
        features (list): List of feature column names.
        threshold (float): Correlation threshold above which one feature is dropped.
        
    Returns:
        list: Refined list of features after removing highly correlated ones.
    """
    corr_matrix = df[features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            if upper_triangle.loc[row, col] > threshold:
                to_drop.add(col)
    refined_features = [f for f in features if f not in to_drop]
    print(f"removed {len(to_drop)} highly correlated features: {to_drop}")
    return refined_features

#############################################
# Low-Value Feature Filtering
#############################################
def remove_low_value_features(df, features, mi_scores, mi_cutoff=0.1, shap_importances=None, shap_cutoff=None):
    """
    Remove features with low MI (or low SHAP if provided).
    
    Args:
        df (DataFrame): The dataset containing features.
        features (list): List of feature column names.
        mi_scores (dict): Mutual information scores keyed by feature name.
        mi_cutoff (float): Minimum MI score to retain a feature.
        shap_importances (dict, optional): SHAP values keyed by feature name.
        shap_cutoff (float, optional): Minimum SHAP importance to retain a feature.
        
    Returns:
        list: Refined list of features after filtering.
    """
    # Filter by mutual information
    keep_by_mi = [f for f in features if mi_scores.get(f, 0) >= mi_cutoff]
    dropped_by_mi = set(features) - set(keep_by_mi)
    print(f"dropped {len(dropped_by_mi)} features with MI < {mi_cutoff}: {dropped_by_mi}")
    
    # Optionally filter by SHAP importance
    if shap_importances is not None and shap_cutoff is not None:
        keep_by_shap = [f for f in keep_by_mi if shap_importances.get(f, 0) >= shap_cutoff]
        dropped_by_shap = set(keep_by_mi) - set(keep_by_shap)
        print(f"dropped {len(dropped_by_shap)} features with SHAP < {shap_cutoff}: {dropped_by_shap}")
        final_features = keep_by_shap
    else:
        final_features = keep_by_mi

    return final_features

#############################################
# Main Refinement Function
#############################################
def main():
    engineered_csv = "data/interim/engineered_features.csv"
    if not os.path.exists(engineered_csv) or os.path.getsize(engineered_csv) == 0:
        sys.exit("Error: engineered features CSV is missing or empty. Run feature engineering first.")
    
    df = pd.read_csv(engineered_csv, parse_dates=["date"])
    print(f"Loaded {len(df)} rows from {engineered_csv}.")
    
    # Generate future return targets for multi-day forecasting
    df = add_future_return_targets(df)
    
    # Identify candidate feature cols (exclude date, close, and target columns)
    all_cols = df.columns.tolist()
    non_feature_cols = ["date", "close", "next_day_return", "day_after_next_return", "future_avg_return"]
    candidate_features = [c for c in all_cols if c not in non_feature_cols]
    # Keep only numeric features for MI computation
    feature_cols = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]
    
    # Compute mutual information scores using 'future_avg_return' as the target
    X = df[feature_cols]
    y = df["future_avg_return"]
    mi_scores_array = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores_array, index=X.columns).to_dict()
    print("Computed Mutual Information Scores:")
    for feature, score in mi_scores.items():
        print(f"  {feature}: {score:.4f}")
    
    # Remove highly correlated features
    refined_features = remove_correlated_features(df, feature_cols, threshold=0.99)
    
    # Remove low-value features based on MI scores
    refined_features = remove_low_value_features(df, refined_features, mi_scores, mi_cutoff=0.1)
    
    print(f"Final feature set has {len(refined_features)} features: {refined_features}")
    
    # Create the final refined dataset (keep date and target along with selected features)
    keep_cols = ["date", "future_avg_return"] + refined_features
    refined_df = df[keep_cols].copy()
    output_csv = "data/processed/refined_features.csv"
    refined_df.to_csv(output_csv, index=False)
    print(f"Refined dataset saved to {output_csv}")

if __name__ == "__main__":
    main()