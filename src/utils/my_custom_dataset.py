import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    """
    A minimal custom dataset to load your time-series data from a CSV file.
    Adjust the code to match your cols (X) and target (y).
    """
    def __init__(self, csv_file, seq_length=30):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.seq_length = seq_length

        # Example: Suppose your refined CSV has columns: [date, future_avg_return, share, open, ...]
        # We want to treat 'future_avg_return' as the target, and everything else as features (minus 'date').

        # 1. Drop non-feature cols you don't want in X
        df_features = self.df.drop(columns=["date", "future_avg_return"])
        self.X_all = df_features.values  # shape: [num_samples, num_features]
        
        # 2. Extract the target
        self.y_all = self.df["future_avg_return"].values  # shape: [num_samples,]

        # 3. Optionally reshape X into sequences. This is where you define how you chunk data.
        # If your data is truly time-series, you might do a rolling window approach.
        # For simplicity, here's a placeholder that doesn't chunk the data.
        self.X_seq = self.X_all  # shape: [num_samples, num_features]
        self.y_seq = self.y_all

        # If you need a rolling window approach, you'd implement that logic here.

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        x = self.X_seq[idx]
        y = self.y_seq[idx]
        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # If you need [seq_length, features], reshape x accordingly.
        # x = x.view(seq_length, num_features)
        return x, y
