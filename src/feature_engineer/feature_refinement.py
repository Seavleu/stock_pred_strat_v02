'''
Instead of subtracting today's close from tomorrow's close,
we can calculate the percentage change in the close price. This 
will allow us to compare returns across different stocks levels and 
evaluating profitability(%)
'''

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

class FeatureRefiner:
    def __init__(self, config):
        self.config = config
    
    def load_engineered_features(self):
        return pd.read_csv(self.config['paths']['engineered_features'])
    
    def drop_highly_correlated(self, df):
        numeric_df = df.select_dtypes(include=[np.number])  # filter only numeric columns
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.config['refinement']['corr_threshold'])]
        df = df.drop(columns=to_drop)
        return df
    
    def drop_low_mutual_information(self, df, target_col):
        features = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        # Clean up infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        target = df[target_col].replace([np.inf, -np.inf], np.nan)
        # Fill NA values with 0
        features = features.fillna(0)
        target = target.fillna(0)
        mi = mutual_info_regression(features, target)
        low_mi = features.columns[mi < self.config['refinement']['mi_threshold']]
        df = df.drop(columns=low_mi) 
        return df
    
    def transform_target(self, df):
        """
        Transform the target variables:
         - Calculate next_day_close, day_after_next_close, and future_5day_close if not present.
         - Compute returns as percentage change relative to current close.
         - Primary target: future_avg_return, the average of next_day_return and day_after_next_return.
        
        This supports our goals:
          • Determine today's entry and tomorrow's exit positions (next_day_return)
          • Determine today's entry and the day after tomorrow's exit positions (day_after_next_return)
          • Determine profitability with a 2-day forecast (future_avg_return)
          • Build long-term strategy with a minimum 5-day forecast
        """
        
        # sort by company & date for proper temporal ordering
        df = df.sort_values(by=['company', 'date']).reset_index(drop=True)
        
        # create future close columns if they don't exist
        if 'next_day_close' not in df.columns:
            df['next_day_close'] = df.groupby('company')['close'].shift(-1)
        if 'day_after_next_close' not in df.columns:
            df['day_after_next_close'] = df.groupby('company')['close'].shift(-2)
        if 'future_5day_close' not in df.columns:
            df['future_5day_close'] = df.groupby('company')['close'].shift(-5)
        
        # compute returns as percentage change from current close
        df['next_day_return'] = (df['next_day_close'] - df['close']) / df['close']
        df['day_after_next_return'] = (df['day_after_next_close'] - df['close']) / df['close']
        df['future_5day_return'] = (df['future_5day_close'] - df['close']) / df['close']
        
        # primary target: average return over next 2 days
        df['future_avg_return'] = (df['next_day_return'] + df['day_after_next_return']) / 2
        
        return df

    def save_refined_features(self, df):
        path = self.config['paths']['refined_features']
        df.to_csv(path, index=False)
        return df

    def run_refinement(self):
        # load engineered features, transform target, drop redundant features, and save results
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
