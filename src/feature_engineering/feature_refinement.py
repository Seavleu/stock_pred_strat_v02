import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

class FeatureRefiner:
    def __init__(self, config):
        self.config = config
    
    def load_engineered_features(self):
        return pd.read_csv(self.config['paths']['engineered_features'])
    
    def drop_highly_correlated(self, df):
        numeric_df = df.select_dtypes(include=[np.number])  # filter only numeric cols
        corr_matrix = numeric_df.corr().abs()  # remove features with correlation above threshold (0.99)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.config['refinement']['corr_threshold'])]
        df = df.drop(columns=to_drop)
        return df
    
    def drop_low_mutual_information(self, df, target_col):
        features = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        
        # clean up inf or too large vals
        features = features.replace([np.inf, -np.inf], np.nan)
        target = df[target_col].replace([np.inf, -np.inf], np.nan)

        # fil NA values with 0
        features = features.fillna(0)
        target = target.fillna(0)

        mi = mutual_info_regression(features, target)
        low_mi = features.columns[mi < self.config['refinement']['mi_threshold']]
        df = df.drop(columns=low_mi) 
        return df
    
    def transform_target(self, df):
        # Sort by company & date to get correct temporal ordering
        df = df.sort_values(by=['company', 'date']).reset_index(drop=True)

        # Create future-close columns if they don't exist
        if 'day_after_next_close' not in df.columns:
            df['day_after_next_close'] = df.groupby('company')['close'].shift(-2)
        if 'future_5day_close' not in df.columns:
            df['future_5day_close'] = df.groupby('company')['close'].shift(-5)

        if 'next_day_close' not in df.columns:
            df['next_day_close'] = df.groupby('company')['close'].shift(-1)

        # Compute future_avg_return
        df['future_avg_return'] = ((df['next_day_close'] + df['day_after_next_close']) / 2) - df['close']
        return df

    
    def save_refined_features(self, df):
        path = self.config['paths']['refined_features']
        df.to_csv(path, index=False)
        return df
    # we have to run it in order loaded data -> transform -> drop SHAP -> drop LMI on target_col -> save file
    def run_refinement(self):
        df = self.load_engineered_features()
        df = self.transform_target(df) 
        df = self.drop_highly_correlated(df)
        df = self.drop_low_mutual_information(df, target_col='future_avg_return')
        df = self.save_refined_features(df)
        return df


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    refiner = FeatureRefiner(config)
    refined_df = refiner.run_refinement()
    print("Feature refinement completed. Refined data saved to:", config['paths']['refined_features'])
