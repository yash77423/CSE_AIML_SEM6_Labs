import torch

# Define x, y, z as tensors with requires_grad=True for gradient computation
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(1.0, requires_grad=True)

# Define the function f(x, y, z)
u = (2 * x) / torch.sin(y)
v = z * u
w = 1 + v
t = torch.log(w)
f = torch.tanh(t)

# Compute the gradients df/dx and df/dy
f.backward()

# Print the computed gradients
print("Computed Gradient df/dx:", x.grad.item())
print("Computed Gradient df/dy:", y.grad.item())

# Now calculate the analytical gradients (as derived above)
sech2_t = 1 - torch.tanh(t)**2  # Computing sech(t)
analytical_df_dx = sech2_t * (1 / w) * z * (2 / torch.sin(y))
analytical_df_dy = sech2_t * (1 / w) * z * (-2 * x * torch.cos(y) / torch.sin(y)**2)

print("Analytical Gradient df/dx:", analytical_df_dx.item())
print("Analytical Gradient df/dy:", analytical_df_dy.item())