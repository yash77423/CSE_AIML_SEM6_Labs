import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Read the data
df = pd.read_csv("./data/NaturalGasPrice/daily.csv")

# Preprocess the data: Drop NA values
df = df.dropna()

# Normalize the prices
y = df['Price'].values
x = np.arange(1, len(y) + 1, 1)

# Normalizing the prices between 0 and 1
minm = y.min()
maxm = y.max()
y = (y - minm) / (maxm - minm)

# Sequence length is 10 (the last 10 days of price)
Sequence_Length = 10

# Prepare the dataset (X contains sequences, Y is the corresponding next day's price)
X = []
Y = []
for i in range(0, len(y) - Sequence_Length):  # Don't go beyond the data length
    X.append(y[i:i + Sequence_Length])  # Sequence of 10 prices
    Y.append(y[i + Sequence_Length])  # Next day's price

# Convert X, Y to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets (90% train, 10% test)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42, shuffle=False)


# Define a custom Dataset class
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).view(-1, Sequence_Length, 1)  # Shape (batch, sequence_length, 1)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Shape (batch, 1)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


# Create the training dataset and dataloader
train_dataset = NGTimeSeries(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        # Pass the input through the RNN layer
        output, _status = self.rnn(x)
        # We take the output of the last time step
        output = output[:, -1, :]
        # Pass it through a fully connected layer
        output = self.fc1(torch.relu(output))
        return output


# Initialize the model
model = RNNModel()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 1500
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        # Forward pass
        inputs, targets = data
        y_pred = model(inputs).view(-1)

        # Compute loss
        loss = criterion(y_pred, targets.view(-1))
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluate on the test set
test_dataset = NGTimeSeries(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model.eval()
test_pred = []
test_true = []

with torch.no_grad():
    for data in test_loader:
        inputs, targets = data
        y_pred = model(inputs).view(-1)
        test_pred.extend(y_pred.numpy())
        test_true.extend(targets.numpy())

# Denormalize the predictions and true values
test_pred = np.array(test_pred) * (maxm - minm) + minm
test_true = np.array(test_true) * (maxm - minm) + minm

# Plot the predicted vs true values
plt.plot(test_pred, label='Predicted')
plt.plot(test_true, label='True')
plt.legend()
plt.show()

# Plot the full series including predictions
plt.plot(y * (maxm - minm) + minm, label='Original Price')
plt.plot(range(len(y) - len(test_pred), len(y)), test_pred, label='Predicted Price')
plt.legend()
plt.show()
