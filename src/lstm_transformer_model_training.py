# src/model_training.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr
import optuna
import joblib
import os

try:
    from tqdm.keras import TqdmCallback
    use_tqdm = True
except ImportError:
    use_tqdm = False

# ---------------------------
# Data Preparation Functions
# ---------------------------
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def compute_ic(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def compute_rank_ic(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def compute_sharpe_ratio(returns):
    std = np.std(returns)
    return np.mean(returns) / std if std != 0 else np.nan

# ---------------------------
# LSTM Model Definition
# ---------------------------
def build_lstm_model(input_shape, params):
    model = Sequential()
    model.add(LSTM(params['lstm_units'], input_shape=input_shape, return_sequences=True))
    model.add(Dropout(params['dropout']))
    model.add(LSTM(params['lstm_units']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

# ---------------------------
# Positional Encoding Layer
# ---------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'pos_encoding': self.pos_encoding})
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# ---------------------------
# Vanilla Transformer Model
# ---------------------------
def build_transformer_model(input_shape, params):
    seq_length, num_features = input_shape
    d_model = params['d_model']
    
    inputs = Input(shape=(seq_length, num_features))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(seq_length, d_model)(x)
    
    attn_output = MultiHeadAttention(num_heads=params['num_heads'], key_dim=d_model)(x, x)
    x = layers.Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ffn = Sequential([
        layers.Dense(params['dff'], activation='relu'),
        layers.Dense(d_model)
    ])
    ffn_output = ffn(x)
    x = layers.Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

# ---------------------------
# Objective Function for Optuna
# ---------------------------
def objective(trial, X_train, y_train, X_val, y_val, model_type, input_shape):
    if model_type == 'lstm':
        params = {
            'lstm_units': trial.suggest_int('lstm_units', 16, 128),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        model = build_lstm_model(input_shape, params)
    elif model_type == 'transformer':
        params = {
            'd_model': trial.suggest_int('d_model', 32, 128),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'dff': trial.suggest_int('dff', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        model = build_transformer_model(input_shape, params)
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# ---------------------------
# Main Training Pipeline
# ---------------------------
def run_training_pipeline(config):
    # Load dynamic selected features dataset
    df = pd.read_csv(config['paths']['dynamic_features'])
    target = 'future_avg_return'
    feature_cols = [col for col in df.columns if col not in ['date', 'company', 'ticker', target]]
    
    # Sort by date for chronological integrity
    data = df.sort_values('date')
    X = data[feature_cols].values
    y = data[target].values
    
    seq_length = config['training'].get('sequence_length', 10)
    X_seq, y_seq = create_sequences(X, y, seq_length)
    
    total_samples = X_seq.shape[0]
    train_end = int(total_samples * 0.8)
    val_end = int(total_samples * 0.9)
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    model_type = config['training'].get('model_type', 'lstm')
    input_shape = X_train.shape[1:]  # (seq_length, num_features)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, model_type, input_shape),
                   n_trials=config['training'].get('n_trials', 10))
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    
    if model_type == 'lstm':
        final_model = build_lstm_model(input_shape, best_params)
    elif model_type == 'transformer':
        final_model = build_transformer_model(input_shape, best_params)
    
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    
    final_epochs = config['training'].get('final_epochs', 20)
    
    # Prepare callbacks: EarlyStopping, ModelCheckpoint, and Tqdm logging if available
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=config['paths']['model'], monitor='val_loss', save_best_only=True)
    ]
    if use_tqdm:
        callbacks.append(TqdmCallback(verbose=1))
    
    final_model.fit(
        X_train_val, y_train_val,
        validation_data=(X_val, y_val),
        epochs=final_epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=0  # tqdm callback will handle progress bar if available
    )
    
    # Save the final model (ModelCheckpoint already saved best, but ensure directory exists)
    os.makedirs(os.path.dirname(config['paths']['model']), exist_ok=True)
    final_model.save(config['paths']['final_model'])
    
    preds = final_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    ic = compute_ic(y_test, preds.flatten())
    rank_ic = compute_rank_ic(y_test, preds.flatten())
    
    print("Evaluation Metrics:")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)
    print("MAPE:", mape)
    print("IC:", ic)
    print("Rank IC:", rank_ic)
    
    signals = np.where(preds.flatten() > 0, 1, -1)
    strategy_returns = signals * y_test
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    sharpe = compute_sharpe_ratio(strategy_returns)
    print("Sharpe Ratio:", sharpe)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'IC': ic,
        'Rank_IC': rank_ic,
        'Sharpe_Ratio': sharpe
    }
    os.makedirs(os.path.dirname(config['paths']['metrics']), exist_ok=True)
    joblib.dump(metrics, config['paths']['metrics'])
    
    return final_model, metrics

if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_training_pipeline(config)
