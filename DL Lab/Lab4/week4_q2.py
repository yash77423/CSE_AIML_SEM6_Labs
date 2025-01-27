# Import necessary Libraries
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

# Initialize loss list and seed
loss_list = []
torch.manual_seed(42)

# Step 1: Initialize inputs and expected outputs as per the truth table of XOR

# Create the tensors x1, x2 and y.
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)


# Step 2: Define XORModel class - write constructor and forward function

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Define the layers
        self.linear1 = nn.Linear(2, 2, bias=True)  # First layer: 2 input -> 2 hidden neurons
        self.activation1 = nn.ReLU()  # ReLU activation for hidden layer
        self.linear2 = nn.Linear(2, 1, bias=True)  # Second layer: 2 hidden neurons -> 1 output
        self.activation2 = nn.Sigmoid()  # Sigmoid activation for output layer (output should be between 0 and 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)  # Apply ReLU activation
        x = self.linear2(x)
        x = self.activation2(x)  # Apply Sigmoid activation for output
        return x


# Step 3: Create DataLoader. Write Dataset class with necessary constructors and methods

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Create the dataset
full_dataset = MyDataset(X, Y)
batch_size = 1

# Create the dataloaders for reading data
train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# Find if CUDA is available to load the model and device onto the available device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model to the selected device
model = XORModel().to(device)
print(model)

# Add the criterion which is the MSELoss
loss_fn = torch.nn.MSELoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)


# Step 4: Training function for one epoch
def train_one_epoch(epoch_index):
    total_loss = 0.0
    # Iterate through the data in the data loader
    for i, data in enumerate(train_data_loader):
        inputs, labels = data

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs.to(device))

        # Compute the Loss and its gradients
        loss = loss_fn(outputs.flatten(), labels.to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        total_loss += loss.item()
    return total_loss / len(train_data_loader)


# Step 5: Train for a set number of epochs
EPOCHS = 10000
for epoch in range(EPOCHS):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)
    loss_list.append(avg_loss)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}/{EPOCHS}, Loss: {avg_loss}')

# Step 6: Model Inference
for param in model.named_parameters():
    print(param)

# Model inference - similar to prediction in ML
input = torch.tensor([0, 1], dtype=torch.float32).to(device)
model.eval()
print("The input is = {}".format(input))
print("Output y predicted ={}".format(model(input)))

# Display the loss plot
plt.plot(loss_list)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
