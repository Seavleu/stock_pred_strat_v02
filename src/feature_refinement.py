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

def remove_correlated_features(df, features, threshold=0.9):
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
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper_triangle.columns:
        # If any correlation is above threshold, mark one for removal
        for row in upper_triangle.index:
            if upper_triangle.loc[row, col] > threshold:
                # Keep the feature with higher MI or let the user decide
                # For now, drop 'col'
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
        mi_scores (Series): Mutual information scores keyed by feature name.
        mi_cutoff (float): Minimum MI score to keep a feature.
        shap_importances (Series or dict, optional): SHAP values keyed by feature name.
        shap_cutoff (float, optional): Minimum SHAP importance to keep a feature.
        
    Returns:
        list: Refined list of features after removing low-value features.
    """
    # 1. Filter by mutual information
    keep_by_mi = [f for f in features if mi_scores.get(f, 0) >= mi_cutoff]
    dropped_by_mi = set(features) - set(keep_by_mi)
    print(f"Dropped {len(dropped_by_mi)} features with MI < {mi_cutoff}: {dropped_by_mi}")
    
    # 2. Optionally filter by SHAP importance
    if shap_importances is not None and shap_cutoff is not None:
        keep_by_shap = [f for f in keep_by_mi if shap_importances.get(f, 0) >= shap_cutoff]
        dropped_by_shap = set(keep_by_mi) - set(keep_by_shap)
        print(f"Dropped {len(dropped_by_shap)} features with SHAP < {shap_cutoff}: {dropped_by_shap}")
        final_features = keep_by_shap
    else:
        final_features = keep_by_mi

    return final_features

def main():
    # 1. Load the feature-selected dataset (output of model_optimization.py or earlier step).
    selected_csv = "data/processed/selected_features.csv"
    if not os.path.exists(selected_csv) or os.path.getsize(selected_csv) == 0:
        sys.exit(f"Error: {selected_csv} is missing or empty. Please run model_optimization.py or feature selection first.")
    
    df = pd.read_csv(selected_csv, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from {selected_csv}.")
    
    # 2. Identify which columns are features vs. target/timestamp
    all_cols = df.columns.tolist()
    non_feature_cols = ["timestamp", "closing_price"]
    feature_cols = [c for c in all_cols if c not in non_feature_cols]
    
    # 3. (Optional) Load or define your mutual information scores or SHAP importances
    #    For demonstration, we assume you have a CSV or dictionary with MI scores.
    #    In practice, you might pass them from model_optimization.py.
    
    # Example: Hard-coded or loaded from a file. Must match your actual features.
    mi_scores = {
        # "feature_name": score
        # e.g., "highest_price": 3.14, "lowest_price": 3.14, ...
    }
    
    # If you have them in a CSV:
    # mi_df = pd.read_csv("data/processed/mi_scores.csv")
    # mi_scores = pd.Series(mi_df.mi_score.values, index=mi_df.feature_name).to_dict()
    
    # 4. Remove Highly Correlated Features
    refined_features = remove_correlated_features(df, feature_cols, threshold=0.9)
    
    # 5. Remove Low-Value Features (MI < 0.1 by default). Adjust as needed.
    refined_features = remove_low_value_features(
        df,
        refined_features,
        mi_scores,
        mi_cutoff=0.1,
        shap_importances=None,  # If you have SHAP importances, pass them here
        shap_cutoff=None        # e.g., shap_cutoff=0.01
    )
    
    # 6. Create a final refined dataset
    keep_cols = ["timestamp", "closing_price"] + refined_features
    refined_df = df[keep_cols].copy()
    print(f"Final feature set has {len(refined_features)} features.")
    
    # 7. Save the refined dataset
    output_csv = "data/processed/refined_features.csv"
    refined_df.to_csv(output_csv, index=False)
    print(f"Refined dataset saved to {output_csv}")

if __name__ == "__main__":
    main()
