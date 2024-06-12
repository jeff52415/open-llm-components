import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


# Example usage
x = torch.linspace(-5, 5, 100)  # Generate 100 points between -5 and 5
y = swish(x)

# Visualization

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), label="Swish")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Swish Activation Function")
plt.legend()
plt.grid(True)
plt.show()
