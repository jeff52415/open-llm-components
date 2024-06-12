import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sigmoid()
        self.activation = (
            nn.SiLU()
        )  # Sigmoid-weighted Linear Unit (SiLU) is also known as Swish

    def forward(self, x):
        return self.activation(self.linear1(x)) * self.gate(self.linear2(x))


# Example usage
input_dim = 128
hidden_dim = 256

swiglu = SwiGLU(input_dim, hidden_dim)
x = torch.randn(32, input_dim)  # Batch of 32 with input_dim features
output = swiglu(x)

print(output.shape)  # Should be (32, hidden_dim)
