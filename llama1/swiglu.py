import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim * 2)  # Output is 2x hidden_dim
        self.gate = nn.SiLU()  # Swish activation

    def forward(self, x):
        x = self.linear(x)  # Shape: (batch, hidden_dim * 2)
        x1, x2 = x.chunk(2, dim=-1)  # Split into two equal parts
        return x1 * self.gate(x2)  # Element-wise product

# Example usage
input_dim = 128
hidden_dim = 256

swiglu = SwiGLU(input_dim, hidden_dim)
x = torch.randn(32, input_dim)  # Batch of 32 with input_dim features
output = swiglu(x)

print(output.shape)  # Should be (32, hidden_dim)