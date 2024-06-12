import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[: x.size(1), :].permute(1, 0, 2)
        else:
            x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def visualize_positional_encodings(encoder, max_len=100):
    # Get the positional encodings
    positional_encodings = encoder.pe[:max_len].squeeze(1).detach().cpu().numpy()

    # Plot the positional encodings
    plt.figure(figsize=(15, 5))
    plt.imshow(positional_encodings, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Positional Encodings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.show()


# Define model parameters
d_model = 512
max_len = 100

# Create the positional encoder
encoder = SinusoidalPositionalEncoder(
    d_model=d_model, max_len=max_len, batch_first=True
)

# Visualize the positional encodings
visualize_positional_encodings(encoder, max_len)
