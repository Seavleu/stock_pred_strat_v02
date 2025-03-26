import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, Sequential, mixed_precision
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr
import optuna
import joblib
import logging

from models.layers.encoding import PositionalEncoding

try:
    from tqdm.keras import TqdmCallback
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ---------------------------------------------------------------------------
# Configure GPU, Mixed Precision, and Logging
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs are available and memory growth is enabled.")
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled: using 'mixed_float16' policy.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Preparation Functions
# ---------------------------------------------------------------------------
def create_sequences(X, y, seq_length):
    x_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        x_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(x_seq), np.array(y_seq)

def compute_ic(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def compute_rank_ic(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def compute_sharpe_ratio(returns):
    std = np.std(returns)
    return np.mean(returns) / std if std != 0 else np.nan

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
def build_lstm_model(input_shape, params):
    model = Sequential(name="lstm_model")
    model.add(LSTM(params['lstm_units'], input_shape=input_shape, return_sequences=True, name="lstm_layer_1"))
    model.add(Dropout(params['dropout'], name="dropout_1"))
    model.add(LSTM(params['lstm_units'], name="lstm_layer_2"))
    model.add(Dropout(params['dropout'], name="dropout_2"))
    model.add(Dense(1, name="dense_output"))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

def build_transformer_model(input_shape, params):
    seq_length, num_features = input_shape
    d_model = params['d_model']
    inputs = Input(shape=(seq_length, num_features), name="transformer_input")
    x = layers.Dense(d_model, name="dense_projection")(inputs)
    x = PositionalEncoding(seq_length, d_model)(x)
    attn_output = MultiHeadAttention(num_heads=params['num_heads'], key_dim=d_model, name="multihead_attention")(x, x)
    x = layers.Add(name="add_attention")([x, attn_output])
    x = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x)
    ffn = Sequential([
        layers.Dense(params['dff'], activation='relu', name="ffn_dense_1"),
        layers.Dense(d_model, name="ffn_dense_2")
    ], name="ffn")
    ffn_output = ffn(x)
    x = layers.Add(name="add_ffn")([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x)
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)
    outputs = Dense(1, dtype='float32', name="dense_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="vanilla_transformer")
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
    return model

def build_informer_model(input_shape, params):
    logger.info("Building Informer model (placeholder implementation).")
    return build_transformer_model(input_shape, params)

def build_tft_model(input_shape, params):
    logger.info("Building TFT model (placeholder implementation).")
    return build_transformer_model(input_shape, params)

def build_ensemble_model(models, weights=None):
    def ensemble_predict(X):
        predictions = [model.predict(X) for model in models]
        predictions = np.array(predictions)
        if weights is None:
            return np.mean(predictions, axis=0)
        else:
            return np.average(predictions, axis=0, weights=weights)
    return ensemble_predict

# ---------------------------------------------------------------------------
# Objective Function for Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------
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
    elif model_type == 'informer':
        params = {
            'd_model': trial.suggest_int('d_model', 32, 128),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'dff': trial.suggest_int('dff', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        model = build_informer_model(input_shape, params)
    elif model_type == 'tft':
        params = {
            'd_model': trial.suggest_int('d_model', 32, 128),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'dff': trial.suggest_int('dff', 64, 256),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        }
        model = build_tft_model(input_shape, params)
    else:
        raise ValueError("Unknown model type: {}".format(model_type))
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# ---------------------------------------------------------------------------
# Main Advanced Training Pipeline with Resume Capability
# ---------------------------------------------------------------------------
def run_advanced_training_pipeline(config):
    # Load and prepare data
    df = pd.read_csv(config['paths']['dynamic_features'])
    target = 'future_avg_return'
    feature_cols = [col for col in df.columns if col not in ['date', 'company', 'ticker', target]]
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
    input_shape = X_train.shape[1:]
    
    final_model_path = config['paths']['model_informer_final']
    
    # Check if we should resume training from an existing checkpoint.
    if config['training'].get('resume_training', False) and os.path.exists(final_model_path):
        final_model = tf.keras.models.load_model(final_model_path, compile=False)
        logger.info("Resumed training from existing model checkpoint: %s", final_model_path)
        additional_epochs = config['training'].get('additional_epochs', 0)
        if additional_epochs > 0:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ModelCheckpoint(filepath=config['paths']['model_informer'], monitor='val_loss', save_best_only=True)
            ]
            if USE_TQDM:
                callbacks.append(TqdmCallback(verbose=1))
            final_model.fit(
                np.concatenate([X_train, X_val], axis=0), 
                np.concatenate([y_train, y_val], axis=0),
                validation_data=(X_val, y_val),
                epochs=additional_epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
    else:
        # Hyperparameter tuning using Optuna
        logger.info("Starting hyperparameter tuning for model type: %s", model_type)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, model_type, input_shape),
                       n_trials=config['training'].get('n_trials', 10))
        best_params = study.best_params
        logger.info("Best hyperparameters: %s", best_params)
        
        if model_type == 'lstm':
            final_model = build_lstm_model(input_shape, best_params)
        elif model_type == 'transformer':
            final_model = build_transformer_model(input_shape, best_params)
        elif model_type == 'informer':
            final_model = build_informer_model(input_shape, best_params)
        elif model_type == 'tft':
            final_model = build_tft_model(input_shape, best_params)
        
        X_train_val = np.concatenate([X_train, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)
        final_epochs = config['training'].get('final_epochs', 20)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=config['paths']['model_informer'], monitor='val_loss', save_best_only=True)
        ]
        if USE_TQDM:
            callbacks.append(TqdmCallback(verbose=1))
        final_model.fit(
            X_train_val, y_train_val,
            validation_data=(X_val, y_val),
            epochs=final_epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
    
    # Ensure directory exists and save the final model
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    final_model.save(final_model_path)
    
    preds = final_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    ic = compute_ic(y_test, preds.flatten())
    rank_ic = compute_rank_ic(y_test, preds.flatten())
    
    logger.info("Evaluation Metrics:")
    logger.info("MSE: %f", mse)
    logger.info("RMSE: %f", rmse)
    logger.info("R²: %f", r2)
    logger.info("MAPE: %f", mape)
    logger.info("IC: %f", ic)
    logger.info("Rank IC: %f", rank_ic)
    
    signals = np.where(preds.flatten() > 0, 1, -1)
    strategy_returns = signals * y_test
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    sharpe = compute_sharpe_ratio(strategy_returns)
    logger.info("Sharpe Ratio: %f", sharpe)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'IC': ic,
        'Rank_IC': rank_ic,
        'Sharpe_Ratio': sharpe
    }
    os.makedirs(os.path.dirname(config['paths']['metrics_informer']), exist_ok=True)
    joblib.dump(metrics, config['paths']['metrics_informer'])
    
    return final_model, metrics

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    start_time = time.time()
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model, metrics = run_advanced_training_pipeline(config)
    elapsed = time.time() - start_time
    logger.info("Advanced training pipeline completed in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60.0)
