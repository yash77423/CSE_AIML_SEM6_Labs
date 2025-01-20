import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Step 1: Define the RegressionModel as a subclass of nn.Module
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Define the parameters w (weight) and b (bias) as torch tensors
        self.w = nn.Parameter(torch.tensor([1.0]))  # Initialize weight to 1.0
        self.b = nn.Parameter(torch.tensor([1.0]))  # Initialize bias to 1.0

    def forward(self, x):
        # Implement the forward pass: y = wx + b
        return self.w * x + self.b


# Step 2: Create a custom Dataset class to handle the data
class RegressionDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Step 3: Prepare the data
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

# Create the dataset and dataloader
dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Step 4: Set up the model, loss function, and optimizer
model = RegressionModel()
criterion = nn.MSELoss()  # MSE Loss function
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent optimizer

# Step 5: Train the model for 100 epochs
loss_list = []

for epoch in range(100):
    total_loss = 0.0

    # Loop through the data in batches using the DataLoader
    for x_batch, y_batch in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute the predicted values
        y_pred = model(x_batch)

        # Compute the loss
        loss = criterion(y_pred, y_batch)

        # Backward pass: compute gradients
        loss.backward()

        # Update the parameters using the optimizer
        optimizer.step()

        # Accumulate the total loss for this batch
        total_loss += loss.item()

    # Calculate the average loss for this epoch
    avg_loss = total_loss / len(dataloader)

    # Append the average loss for plotting
    loss_list.append(avg_loss)

    # Print the parameters and loss
    print(f"Epoch [{epoch + 1}/100], w = {model.w.item():.4f}, b = {model.b.item():.4f}, Loss = {avg_loss:.4f}")

# Step 6: Plot the loss over epochs
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()
