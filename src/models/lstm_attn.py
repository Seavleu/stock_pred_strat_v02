import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A basic LSTM model for sequence-to-one forecasting.
    If you need attention-based LSTM, you can expand this class with an attention mechanism.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        out, _ = self.lstm(x)         # [batch_size, seq_length, hidden_size]
        out = out[:, -1, :]           # take the last timestep
        out = self.fc(out)            # [batch_size, output_size]
        return out
