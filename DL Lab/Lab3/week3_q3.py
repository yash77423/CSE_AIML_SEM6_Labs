import torch
from matplotlib import pyplot as plt

class RegressionModel:
    def __init__(self):
        # Initialize the weights and bias as tensors with requires_grad set to True
        self.w = torch.tensor([1.0], requires_grad=True)  # w initialized to 1
        self.b = torch.tensor([1.0], requires_grad=True)  # b initialized to 1

    def forward(self, x):
        # Perform the forward pass: wx + b
        return self.w * x + self.b

    def update(self, learning_rate):
        # Update the weight and bias based on the gradients
        with torch.no_grad():
            self.w -= learning_rate * self.w.grad
            self.b -= learning_rate * self.b.grad

    def reset_grad(self):
        # Reset gradients to zero
        self.w.grad.zero_()
        self.b.grad.zero_()

    def criterion(self, y, yp):
        # Compute Mean Squared Error (MSE) loss
        return ((y - yp) ** 2).mean()

# Given data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
learning_rate = 0.001  # Learning rate

# Initialize model
model = RegressionModel()

# List to track loss values for plotting
loss_list = []

# Run the training loop for 100 epochs
for epoch in range(100):
    total_loss = 0.0

    # Loop over the data points
    for j in range(len(x)):
        # Forward pass: compute predicted y
        y_pred = model.forward(x[j])

        # Compute the loss (MSE)
        loss = model.criterion(y[j], y_pred)

        # Accumulate the loss
        total_loss += loss

        # Backward pass: compute gradients
        loss.backward()

    # Average loss for this epoch
    avg_loss = total_loss / len(x)

    # Append loss to loss_list for plotting
    loss_list.append(avg_loss.item())

    # Update model parameters using gradient descent
    model.update(learning_rate)

    # Reset gradients for the next iteration
    model.reset_grad()

    # Print the parameters and loss
    print(f"Epoch [{epoch+1}/100], w = {model.w.item():.4f}, b = {model.b.item():.4f}, Loss = {avg_loss.item():.4f}")

# Plot the loss vs epoch graph
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()
