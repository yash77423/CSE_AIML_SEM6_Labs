import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data
X = np.array([[3, 8], [4, 5], [5, 7], [6, 3], [2, 1]], dtype=np.float32)
Y = np.array([-3.7, 3.5, 2.5, 11.5, 5.7], dtype=np.float32)

# Convert to PyTorch tensors
X_train = torch.tensor(X)
Y_train = torch.tensor(Y).view(-1, 1)  # Reshaping Y for the model


# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2 input features and 1 output

    def forward(self, x):
        return self.linear(x)


# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    # Forward pass
    Y_pred = model(X_train)

    # Compute the loss
    loss = criterion(Y_pred, Y_train)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model with a new data point (X1=3, X2=2)
model.eval()  # Set the model to evaluation mode
test_input = torch.tensor([[3, 2]], dtype=torch.float32)
predicted_output = model(test_input).item()

print(f"Predicted Y for X1=3, X2=2: {predicted_output:.4f}")
