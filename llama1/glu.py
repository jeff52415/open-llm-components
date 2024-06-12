import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# GLU(x)=(xW1+b1)⊙σ(xW2+b2)
class GLU(nn.Module):
    def __init__(self, input_dim):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        # Split the linear transformation into two parts
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


# Example usage and visualization
input_dim = 1
glu = GLU(input_dim)

# Generate some example data
x = torch.linspace(-5, 5, 100).reshape(-1, 1)
with torch.no_grad():
    y = glu(x).numpy()
