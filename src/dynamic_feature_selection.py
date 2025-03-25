'''
Ensure only potential high value features make it into the final model training step;
In order to help prevent feature explosion, I implement a top-ratio-selection to select
only the top ratio of 50% by importance.

Note: 50% will be a stable clean feature set for the baseline model, but we could adjust it to 70% to capture more specific trend
and 30% for better market state predictions.
'''

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import logging
from tqdm import tqdm

class DynamicFeatureSelector:
    def __init__(self, config):
        self.config = config
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    def load_refined_features(self):
        logging.info("Loading refined feature set...")
        return pd.read_csv(self.config['paths']['refined_features'])

    def rolling_shap_selection(self, df):
        window_size = self.config['dynamic_selection']['window_size']
        step_size = self.config['dynamic_selection']['step_size']
        target_col = 'future_avg_return'
        features = [col for col in df.columns if col not in ['date', 'company', 'ticker', target_col]]

        feature_importance = {}
        num_windows = len(range(0, len(df) - window_size + 1, step_size))

        logging.info(f"Running rolling SHAP selection with window size {window_size}, step size {step_size}, total windows: {num_windows}")

        for i in tqdm(range(0, len(df) - window_size + 1, step_size), desc="Rolling SHAP"):
            window = df.iloc[i:i+window_size]
            X = window[features].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = window[target_col].replace([np.inf, -np.inf], np.nan).fillna(0)

            model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            for idx, feature in enumerate(features):
                imp = np.mean(np.abs(shap_values[:, idx]))
                feature_importance[feature] = feature_importance.get(feature, 0) + imp

            logging.info(f"Window {i}-{i+window_size} processed.")

        # Average importance
        for feature in feature_importance:
            feature_importance[feature] /= num_windows

        # Filter by minimum importance
        min_shap_importance = self.config['dynamic_selection'].get('min_shap_importance', 0.0)
        filtered_importance = {f: imp for f, imp in feature_importance.items() if imp >= min_shap_importance}
        sorted_features = sorted(filtered_importance.items(), key=lambda x: x[1], reverse=True)

        # Select top features
        select_ratio = self.config['dynamic_selection'].get('select_ratio', 0.5)
        num_select = int(len(sorted_features) * select_ratio)
        selected_features = [f for f, _ in sorted_features[:num_select]]

        logging.info(f"Selected top {num_select} features out of {len(features)}")

        meta_cols = ['date', 'company', 'ticker', 'next_day_close', 'day_after_next_close', 'future_5day_close', 'future_avg_return']
        df_selected = df[meta_cols + selected_features]
        return df_selected

    def save_dynamic_features(self, df):
        path = self.config['paths']['dynamic_features']
        df.to_csv(path, index=False)
        logging.info(f"Dynamic features saved to: {path}")
        return df

    def run_dynamic_selection(self):
        df = self.load_refined_features()
        df_selected = self.rolling_shap_selection(df)
        df_selected = self.save_dynamic_features(df_selected)
        return df_selected

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    selector = DynamicFeatureSelector(config)
    selected_df = selector.run_dynamic_selection() 
    print("Dynamic feature selection completed. Selected features saved to:", config['paths']['dynamic_features'])
