import torch

# Data points
x = torch.tensor([2.0, 4.0])
y = torch.tensor([20.0, 40.0])

# Initialize parameters w and b
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Learning rate
alpha = 0.001


# Function to compute model predictions
def predict(x, w, b):
    return w * x + b


# Perform gradient descent for 2 epochs
for epoch in range(2):
    # Forward pass: compute predictions
    y_pred = predict(x, w, b)

    # Compute the loss (Mean Squared Error)
    loss = torch.mean((y_pred - y) ** 2)

    # Zero the gradients from the previous step
    w.grad = None
    b.grad = None

    # Backward pass: compute gradients
    loss.backward()

    # Print gradients and parameter updates
    print(f"Epoch {epoch + 1}:")
    print(f"  w.grad = {w.grad.item():.6f}, b.grad = {b.grad.item():.6f}")

    # Update parameters using gradient descent
    with torch.no_grad():
        w -= alpha * w.grad
        b -= alpha * b.grad

    # Print updated parameters
    print(f"  Updated w = {w.item():.6f}, b = {b.item():.6f}")
    print(f"  Loss = {loss.item():.6f}\n")

