import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Given data
x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).view(-1, 1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).view(-1, 1)

# Define the linear regression model
model = nn.Linear(1, 1)  # One input feature and one output feature

# Define the loss function (Mean Squared Error) and optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Set the number of epochs
epochs = 1000
losses = []

# Training loop
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute the loss
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    # Backward pass: Compute gradients
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Backpropagation

    # Update the weights using the optimizer
    optimizer.step()

    # Print the loss every 100 epochs for monitoring
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot the loss over epochs
plt.plot(range(epochs), losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss during Linear Regression Training')
plt.legend()
plt.show()
