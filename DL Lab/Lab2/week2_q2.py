import torch

# Define w, x, b as tensors with requires_grad=True for gradient computation
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(1.0)
b = torch.tensor(0.5)

# Define the function a = ReLU(wx + b)
a = torch.relu(w * x + b)

# Compute the gradient da/dw
a.backward()

# Print the computed gradient
print("Computed Gradient da/dw:", w.grad.item())

# Analytical gradient for comparison
# The ReLU derivative is 1 if wx + b > 0, otherwise 0.
wx_b = w * x + b
analytical_gradient = x if wx_b > 0 else 0
print("Analytical Gradient da/dw:", analytical_gradient)