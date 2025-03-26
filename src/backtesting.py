'''
Currently 
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
class BacktestingEngine:
    def __init__(self, config):
        self.config = config
        self.transaction_cost = config['backtesting'].get('transaction_cost', 0.001)  # e.g., 0.1% per trade
        self.slippage = config['backtesting'].get('slippage', 0.0005)  # e.g., 0.05% slippage
        # Dynamic thresholds quantiles: these can be tuned
        self.quantile_up = config['backtesting'].get('threshold_up_quantile', 0.75)
        self.quantile_down = config['backtesting'].get('threshold_down_quantile', 0.25)
    
    def load_data(self):
        """
        Loads the refined/dynamic feature dataset that includes:
         - date, company, ticker,
         - predicted returns (future_avg_return_pred) and actual returns (future_avg_return)
         - open price for next day execution (assumed to be present)
        """
        df = pd.read_csv(self.config['paths']['predictions_backtesting'])
        # Ensure date ordering
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def generate_dynamic_thresholds(self, predictions):
        """
        Compute dynamic thresholds based on historical predictions.
        Here we use quantiles, but you could also use rolling z-scores or other adaptive methods.
        """
        threshold_up = np.quantile(predictions, self.quantile_up)
        threshold_down = np.quantile(predictions, self.quantile_down)
        return threshold_up, threshold_down

    def generate_signals(self, df):
        """
        Generate trading signals:
         - Buy if predicted return > threshold_up
         - Sell if predicted return < threshold_down
         - Hold otherwise
        """
        # Assume the predicted returns column is named "future_avg_return_pred"
        predictions = df['future_avg_return_pred'].values
        threshold_up, threshold_down = self.generate_dynamic_thresholds(predictions)
        signals = []
        for pred in predictions:
            if pred > threshold_up:
                signals.append('Buy')
            elif pred < threshold_down:
                signals.append('Sell')
            else:
                signals.append('Hold')
        df['signal'] = signals
        df['threshold_up'] = threshold_up  # for record keeping
        df['threshold_down'] = threshold_down
        return df

    def simulate_trades(self, df):
        """
        Simulate trade execution:
         - Entry is assumed at the next day's opening price.
         - Incorporate transaction cost and slippage.
         - Calculate daily returns from strategy.
        """
        # Create columns for trade execution.
        # Assume that the dataframe has the following columns:
        # "open" for next day's open price, "future_avg_return" as actual return target.
        # For simplicity, we'll assume we hold a fixed position (1 unit) when in the market.
        
        positions = []  # 1 for long, -1 for short, 0 for hold
        entry_prices = []  # simulated entry price
        trade_returns = []  # return for the trade

        # We simulate day-by-day
        for idx in range(len(df) - 1):
            signal = df.loc[idx, 'signal']
            # Assume trade is executed at next day's open price
            next_open = df.loc[idx+1, 'open']
            # Apply transaction cost and slippage when entering the position
            if signal == 'Buy':
                pos = 1
                effective_entry = next_open * (1 + self.slippage + self.transaction_cost)
            elif signal == 'Sell':
                pos = -1
                effective_entry = next_open * (1 - self.slippage - self.transaction_cost)
            else:
                pos = 0
                effective_entry = np.nan
            positions.append(pos)
            entry_prices.append(effective_entry)
            
            # Calculate trade return on next day's performance: 
            # For simplicity, assume the actual return is given in "future_avg_return"
            actual_return = df.loc[idx, 'future_avg_return']
            trade_ret = pos * actual_return
            trade_returns.append(trade_ret)

        # Align positions with dates (pad last day with 0 position)
        positions.append(0)
        df['position'] = positions
        df['trade_return'] = trade_returns + [0]
        
        # Calculate cumulative strategy return
        df['cumulative_return'] = (1 + df['trade_return']).cumprod() - 1
        return df

    def calculate_performance_metrics(self, df):
        """
        Calculate performance metrics:
         - Regression: MSE, RMSE, R², MAPE (if applicable to predictions)
         - Finance: Sharpe Ratio, IC, Rank IC, Max Drawdown, Win Ratio, Average Holding Period
        """
        # Financial Metrics:
        # 1. Sharpe Ratio based on daily trade returns:
        trade_returns = df['trade_return'].dropna().values
        sharpe = np.mean(trade_returns) / np.std(trade_returns) if np.std(trade_returns) != 0 else np.nan
        
        # 2. Information Coefficient (IC): Pearson correlation between predictions and actual returns
        ic, _ = pearsonr(df['future_avg_return_pred'], df['future_avg_return'])
        # 3. Rank IC (RIC): Spearman correlation
        ric = spearmanr(df['future_avg_return_pred'], df['future_avg_return']).correlation
        
        # 4. Cumulative Return (last value)
        cumulative_return = df['cumulative_return'].iloc[-1]
        
        # 5. Maximum Drawdown:
        cum_returns = df['cumulative_return']
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 6. Win Ratio: Percentage of trades that resulted in positive returns
        win_ratio = np.mean(df['trade_return'] > 0)
        
        # 7. Average Holding Period (if positions change, compute average duration of continuous position)
        holding_periods = []
        current_period = 0
        current_position = df['position'].iloc[0]
        for pos in df['position']:
            if pos == current_position and pos != 0:
                current_period += 1
            else:
                if current_position != 0:
                    holding_periods.append(current_period)
                current_position = pos
                current_period = 1 if pos != 0 else 0
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0

        metrics = {
            'Sharpe_Ratio': sharpe,
            'IC': ic,
            'Rank_IC': ric,
            'Cumulative_Return': cumulative_return,
            'Max_Drawdown': max_drawdown,
            'Win_Ratio': win_ratio,
            'Average_Holding_Period': avg_holding_period
        }
        return metrics

    def plot_performance(self, df):
        """
        Plot cumulative returns over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['cumulative_return'], label='Strategy Cumulative Return')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Backtesting Performance')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        output_dir = self.config['backtesting'].get('output_dir', 'outputs/plots')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'cumulative_return.png'))
        plt.show()

    def run_backtest(self):
        df = self.load_data()
        df = self.generate_signals(df)
        df = self.simulate_trades(df)
        metrics = self.calculate_performance_metrics(df)
        print("Backtesting Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        self.plot_performance(df)
        return df, metrics

# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    backtester = BacktestingEngine(config)
    df_results, performance_metrics = backtester.run_backtest()
