import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Custom Dataset for Time-Series Data
# -------------------------
class StockDataset(Dataset):
    def __init__(self, csv_file, seq_length=30):
        """
        Args:
            csv_file (str): Path to the engineered features CSV.
            seq_length (int): Number of time steps in each input sequence.
        """
        self.data = pd.read_csv(csv_file, parse_dates=['timestamp'])
        self.data.sort_values("timestamp", inplace=True)
         
        self.features = self.data.drop(columns=["timestamp"]).values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_length]
        
        target_index = list(self.data.columns).index("closing_price") - 1  # -1 if timestamp is dropped
        y = self.features[idx + self.seq_length, target_index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -------------------------
# LSTM Model for Time-Series Forecasting
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            output_size (int): Dimensionality of the model output.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Use the last time step's output for prediction
        out = self.fc(out[:, -1, :])
        return out

# -------------------------
# Training Pipeline
# -------------------------
def train_model():
    # Paths and hyperparameters
    data_path = "data/processed/dynamic_alpha_selected_features.csv" #engineered_features(noisy), dynamic_alpha_selected_features.csv
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please run the feature engineering pipeline first.")
    
    seq_length = 30
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.001

    # Determine input_size based on engineered features (excluding timestamp)
    df = pd.read_csv(data_path)
    # Drop the timestamp column; adjust if you have additional non-feature columns.
    input_size = df.drop(columns=["timestamp"]).shape[1]
    
    # Create dataset and dataloader
    dataset = StockDataset(data_path, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the model, loss function, and optimizer
    hidden_size = 50
    num_layers = 2
    output_size = 1  # Predicting the next closing price
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
        
        epoch_loss /= len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    
    # Save the trained model
    model_save_path = "models/lstm_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()

