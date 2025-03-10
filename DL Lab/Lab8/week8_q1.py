import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


# Generate Fibonacci numbers up to a certain limit
def generate_fibonacci(limit):
    fib = [0, 1]
    while len(fib) < limit:
        fib.append(fib[-1] + fib[-2])
    return fib


# Prepare the dataset for RNN (3 input numbers, 1 target output number)
def prepare_data(fib_sequence):
    X = []
    y = []
    for i in range(len(fib_sequence) - 3):
        X.append(fib_sequence[i:i + 3])  # Last 3 numbers as input
        y.append(fib_sequence[i + 3])  # The next number as output
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # Ensure np.float32 for conversion


# Generate a sequence of Fibonacci numbers
fib_sequence = generate_fibonacci(100)

# Prepare data for training
X, y = prepare_data(fib_sequence)

# Check the shapes and ensure the correct data types
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
print(f"First few values of X: {X[:5]}")
print(f"First few values of y: {y[:5]}")

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)  # Ensure X is of dtype float32
y = torch.tensor(y, dtype=torch.float32)

# Reshape input to be (samples, timesteps, features)
X = X.view(X.shape[0], X.shape[1], 1)  # (samples, 3, 1)


# Define the RNN model
class FibonacciRNN(nn.Module):
    def __init__(self):
        super(FibonacciRNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)  # Output layer to predict the next Fibonacci number

    def forward(self, x):
        # Pass the input through the RNN
        rnn_out, _ = self.rnn(x)
        # Get the output of the last timestep
        last_rnn_out = rnn_out[:, -1, :]
        # Pass the last RNN output through the fully connected layer
        out = self.fc(last_rnn_out)
        return out


# Initialize the model
model = FibonacciRNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X)

    # Compute loss
    loss = criterion(outputs.squeeze(), y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model with an example input
model.eval()
test_input = torch.tensor([1.0, 1.0, 2.0]).view(1, 3, 1)  # Input: [1, 1, 2]
predicted_output = model(test_input)
print(f"Predicted next Fibonacci number for input [1, 1, 2]: {predicted_output.item():.4f}")
