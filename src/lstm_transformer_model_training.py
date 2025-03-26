'''
Task: regression, predict future_avg_return
Metrics: MSE, RMSE, R-square, IC, Rank IC, Sharpe
Workflow: Load -> Seq -> Split -> Optuna Tune -> Final Train -> Evaluate
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
import time
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
# Enable GPU memory growth and log GPU info
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs are available and memory growth is enabled.")
    except RuntimeError as e:
        print(e)

# Enable mixed precision training to leverage Tensor Cores on RTX 3080
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled: using 'mixed_float16' policy.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    outputs = Dense(1, dtype='float32')(x)

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
        verbose=0
    )

    os.makedirs(os.path.dirname(config['paths']['model']), exist_ok=True)
    final_model.save(config['paths']['model_final'])

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
    start_time = time.time()

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    final_model, metrics = run_training_pipeline(config)

    # ----- Save & Visualize Predictions -----
    preds = final_model.predict(X_test)
    signals = np.where(preds.flatten() > 0, 1, -1)
    strategy_returns = signals * y_test
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    sharpe = compute_sharpe_ratio(strategy_returns)

    results_df = pd.DataFrame({
        "True": y_test,
        "Predicted": preds.flatten(),
        "Signal": signals,
        "Strategy_Return": strategy_returns,
        "Cumulative_Return": cumulative_returns
    })

    os.makedirs("outputs/plots", exist_ok=True)
    results_df.to_csv(config['paths']['predictions'], index=False)
    logger.info(f"Predictions and strategy saved to: {config['paths']['predictions']}")

    # plot strategies' performance
    plot_results(results_df)

    elapsed = time.time() - start_time
    logger.info("Training completed in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60.0)
    print("Pipeline completed successfully.")
