import torch

# Define a, b as tensors with requires_grad=True for gradient computation
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0)

# Define the function x, y, and z
x = 2*a + 3*b
y = 5*a**2 + 3*b**3
z = 2*x + 3*y

# Compute the gradient dz/da
z.backward()

# Print the computed gradient
print("Computed Gradient dz/da:", a.grad.item())

# Analytical gradient for comparison
analytical_gradient = 4 + 30 * a.item()
print("Analytical Gradient dz/da:", analytical_gradient)