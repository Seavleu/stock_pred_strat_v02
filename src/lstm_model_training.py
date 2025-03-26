'''
Time-Series Regression Model 
Dynamic feature-selected inputs →  LSTM stack → target: future_avg_return
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def load_dynamic_features(self):
        logging.info("Loading dynamic feature set...")
        return pd.read_csv(self.config['paths']['dynamic_features'])

    def prepare_data(self, df):
        target = 'future_avg_return'         # for multi-step: 'next_day_close', 'day_after_next_close', 'future_5day_close                                 
        feature_cols = [col for col in df.columns if col not in ['date', 'company', 'ticker', target]]
        X = df[feature_cols].values
        y = df[target].values
        logging.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features.")
        return X, y

    def time_series_split(self, X, y, n_splits=5):
        logging.info(f"Splitting data into {n_splits} time series folds...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X))

    def build_lstm_model(self, input_shape, params):
        model = Sequential()
        model.add(LSTM(params['lstm_units'], input_shape=input_shape, return_sequences=True))
        model.add(Dropout(params['dropout']))
        model.add(LSTM(params['lstm_units']))
        model.add(Dropout(params['dropout']))
        model.add(Dense(1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss='mse')
        return model

    def train_model(self, X, y, splits):
        train_idx, val_idx = splits[0]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        def objective(trial):
            params = {
                'lstm_units': trial.suggest_int('lstm_units', 16, 128),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            }

            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]), params)
            for epoch in tqdm(range(10), desc="Tuning Epochs", leave=False):
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=1, batch_size=32, verbose=0)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            return rmse

        logging.info("Starting Optuna hyperparameter tuning...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10, show_progress_bar=True)

        best_params = study.best_params
        logging.info(f"Best hyperparameters found: {best_params}")

        # Final model training
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]), best_params)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        logging.info("Training final model with best parameters...")
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       epochs=20, batch_size=32, callbacks=[early_stop])

        self.model.save(self.config['paths']['model_lstm'])
        logging.info(f"Model saved to {self.config['paths']['model_lstm']}")
        return self.model

    def evaluate_model(self, X_test, y_test):
        logging.info("Evaluating model on test set...")
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        # mape = mean_absolute_percentage_error(y_test, preds) #explode MAPE: 3,711,725,036,697.97
        non_zero_mask = y_test != 0
        mape = mean_absolute_percentage_error(y_test[non_zero_mask], preds[non_zero_mask])
        metrics = {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        return metrics

    def run_training_pipeline(self):
        df = self.load_dynamic_features()
        X, y = self.prepare_data(df)
        splits = self.time_series_split(X, y, n_splits=5)

        self.train_model(X, y, splits)

        _, test_idx = splits[-1]
        X_test, y_test = X[test_idx], y[test_idx]

        # remove NaNs only from the test set before evaluation
        mask = ~np.isnan(y_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        metrics = self.evaluate_model(X_test, y_test)
        logging.info(f"Final evaluation metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    trainer = ModelTrainer(config)
    trainer.run_training_pipeline()
    print("Pipeline completed successfully.")


'''
Final evaluation metrics:
{'MSE': 0.028157837458786035, 
'RMSE': np.float64(0.16780297213930997), 
'R2': 0.8266309633802404, 
'MAPE': 3711725036697.9653} 

we trained on a dynamic, hyperparameter-tuned, LSTM-based stock price return predictor with a 
validation curve and strong r-square performance. This will serve as a baseline

Next:
-> Update to multi-step forecasting
-> add classification labels (buy/hold/sell)
-> inject attention mechanism
'''