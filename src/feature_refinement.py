"""
Feature Refinement Script

Removes highly correlated features and drops low-MI (or low-SHAP) features
from your selected feature set. Saves the refined dataset for re-training
and further model optimization.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def remove_correlated_features(df, features, threshold=0.99):
    """
    Remove features that are highly correlated with one another.
    
    Args:
        df (DataFrame): The dataset containing features.
        features (list): List of feature column names to consider.
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
    print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    return refined_features

def remove_low_value_features(df, features, mi_scores, mi_cutoff=0.1, shap_importances=None, shap_cutoff=None):
    """
    Remove features with low MI or low SHAP importance.
    
    Args:
        df (DataFrame): The dataset containing features.
        features (list): List of feature column names to consider.
        mi_scores (dict): Mutual information scores keyed by feature name.
        mi_cutoff (float): Minimum MI score to keep a feature.
        shap_importances (dict, optional): SHAP values keyed by feature name.
        shap_cutoff (float, optional): Minimum SHAP importance to keep a feature.
        
    Returns:
        list: Refined list of features after removing low-value features.
    """
    # Filter by mutual information
    keep_by_mi = [f for f in features if mi_scores.get(f, 0) >= mi_cutoff]
    dropped_by_mi = set(features) - set(keep_by_mi)
    print(f"Dropped {len(dropped_by_mi)} features with MI < {mi_cutoff}: {dropped_by_mi}")
    
    # Optionally filter by SHAP importance if provided
    if shap_importances is not None and shap_cutoff is not None:
        keep_by_shap = [f for f in keep_by_mi if shap_importances.get(f, 0) >= shap_cutoff]
        dropped_by_shap = set(keep_by_mi) - set(keep_by_shap)
        print(f"Dropped {len(dropped_by_shap)} features with SHAP < {shap_cutoff}: {dropped_by_shap}")
        final_features = keep_by_shap
    else:
        final_features = keep_by_mi

    return final_features

def main():
    # 1. Load the feature-selected dataset (from model_optimization.py or similar)
    selected_csv = "data/processed/selected_features.csv"
    if not os.path.exists(selected_csv) or os.path.getsize(selected_csv) == 0:
        sys.exit(f"Error: {selected_csv} is missing or empty. Please run model_optimization.py or feature selection first.")
    
    df = pd.read_csv(selected_csv, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {selected_csv}.")

    # 2. Identify feature columns (exclude timestamp and target 'closing_price')
    all_cols = df.columns.tolist()
    non_feature_cols = ["timestamp", "closing_price"]
    feature_cols = [c for c in all_cols if c not in non_feature_cols]
    
    # 3. Compute mutual information scores for each feature
    X = df[feature_cols]
    y = df["closing_price"]
    mi_scores_array = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores_array, index=X.columns).to_dict()
    print("Computed Mutual Information Scores:")
    for feature, score in mi_scores.items():
        print(f"  {feature}: {score:.4f}")
    
    # 4. Remove Highly Correlated Features using a relaxed threshold
    refined_features = remove_correlated_features(df, feature_cols, threshold=0.99)
    
    # 5. Remove Low-Value Features based on MI scores (use a low cutoff to retain more features)
    refined_features = remove_low_value_features(df, refined_features, mi_scores, mi_cutoff=0.1, shap_importances=None, shap_cutoff=None)
    
    print(f"Final feature set has {len(refined_features)} features: {refined_features}")
    
    # 6. Create and save the final refined dataset
    keep_cols = ["timestamp", "closing_price"] + refined_features
    refined_df = df[keep_cols].copy()
    output_csv = "data/processed/refined_features.csv"
    refined_df.to_csv(output_csv, index=False)
    print(f"Refined dataset saved to {output_csv}")

if __name__ == "__main__":
    main()
