import torch

x = torch.tensor(2.0, requires_grad=True)
# Analytical gradient for comparison
analytical_gradient = 32*x**3 + 9*x**2 + 14*x + 6
print("Analytical Gradient:", analytical_gradient.item())

y=8*x**4+3*x**3+7*x**2+6*x+3
y.backward()
print(x.grad)

