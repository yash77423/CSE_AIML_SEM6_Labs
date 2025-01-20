import torch

x = torch.tensor([2.0])
x.requires_grad_(True)  #indicate we will need the gradients with respect to this variable
y = x**2 + 5
print(y)

y.backward()  #dy/dx
print('PyTorch gradient:', x.grad)

#Let us compare with the analytical gradient of y = x**2+5 with torch.no_grad():
#this is to only use the tensor value without its gradient information
dy_dx = 2*x  #analytical gradient
print('Analytical gradient:',dy_dx)

