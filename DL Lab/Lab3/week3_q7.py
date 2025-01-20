import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data
x = np.array([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=np.float32).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32).reshape(-1, 1)

# Convert to PyTorch tensors
X_train = torch.tensor(x)
Y_train = torch.tensor(y)


# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        # One input feature and one output (binary classification)
        self.linear = nn.Linear(1, 1)  # Linear layer (input size 1, output size 1)

    def forward(self, x):
        # Sigmoid activation to output a probability between 0 and 1
        return torch.sigmoid(self.linear(x))


# Initialize the model, loss function, and optimizer
model = LogisticRegressionModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
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

# Test the model on the training data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to compute gradients during inference
    Y_pred = model(X_train)
    Y_pred = Y_pred.round()  # Convert probabilities to binary labels (0 or 1)

# Print predictions and true values
print("\nPredicted labels: ", Y_pred.flatten().numpy())
print("True labels: ", Y_train.flatten().numpy())
