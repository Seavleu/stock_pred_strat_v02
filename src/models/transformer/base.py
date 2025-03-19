import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_size=1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x: [batch, seq_length, input_size]
        x = self.input_linear(x)         # [batch, seq_length, d_model]
        x = self.pos_encoder(x)          # add positional encoding
        x = x.transpose(0, 1)            # Transformer expects [seq_length, batch, d_model]
        x = self.transformer_encoder(x)  # [seq_length, batch, d_model]
        out = x[-1, :, :]                # take the last time step
        out = self.fc(out)               # [batch, output_size]
        return out
