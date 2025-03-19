# File: src/model_training.py

import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from src.models.transformer.base import TransformerModel
from src.models.lstm_attn import LSTMModel  
from src.utils.my_custom_dataset import MyCustomDataset

def objective(trial, train_loader, input_size, model_type="Transformer"):
    """
    The Optuna objective function that trains either a Transformer or LSTM-based model 
    and returns the final training loss for hyperparameter optimization.
    """
    # Suggest hyperparameters
    seq_length = trial.suggest_int("seq_length", 30, 90)
    d_model = trial.suggest_int("d_model", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model type
    if model_type == "Transformer":
        model = TransformerModel(
            input_size=input_size,
            d_model=d_model,
            nhead=4,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
    else:
        # Example if you have an LSTM model
        model = LSTMModel(input_size, d_model, num_layers, dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Simple training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        trial.report(epoch_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return epoch_loss

def run_optuna_study(train_loader, input_size, model_type="Transformer"):
    """
    Create and run the Optuna study using the objective function above.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_loader, input_size, model_type),
                   n_trials=10)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

def main():
    # Load your dataset (dynamic_selected_features.csv or refined_features.csv)
    dataset = MyCustomDataset("data/dynamic_selected_features.csv", seq_length=30)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Suppose the dataset returns x with shape [batch_size, seq_length, num_features]
    input_size = dataset.num_features
    
    model = LSTMModel(input_size=dataset.X_seq.shape[1], hidden_size=64, num_layers=2, dropout=0.2)

    # Run Optuna study to find best hyperparameters for the Transformer
    best_params = run_optuna_study(train_loader, input_size, model_type="Transformer")

    # Train final model with best hyperparams, etc.
    # ...
    # You can also implement your evaluation and backtesting here.

if __name__ == "__main__":
    main()
