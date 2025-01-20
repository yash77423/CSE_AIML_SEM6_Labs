import torch

# Define x as a tensor with requires_grad=True
x = torch.tensor(1.0, requires_grad=True)

# Define the function f(x) = exp(-x^2 - 2x - sin(x))
f = torch.exp(-x**2 - 2*x - torch.sin(x))

# Compute the gradient df/dx
f.backward()

# Print the computed gradient
print("Computed Gradient df/dx:", x.grad.item())

# Analytical gradient for comparison
import math
analytical_gradient = torch.exp(-x**2 - 2*x - torch.sin(x)) * (-2*x - 2 - torch.cos(x))
print("Analytical Gradient df/dx:", analytical_gradient.item())