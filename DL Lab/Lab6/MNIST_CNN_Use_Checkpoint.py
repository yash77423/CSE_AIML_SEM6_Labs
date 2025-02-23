import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define the CNN Model (MNIST_CNN.py)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define loss function and optimizer
model = CNNClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Check if checkpoint exists
checkpoint_path = './checkpoints/checkpoint_2.pt'  # Example: checkpoint after epoch 2

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    last_epoch = checkpoint['last_epoch']
    last_loss = checkpoint['last_loss']
    print(f"Resuming training from epoch {last_epoch}, last loss: {last_loss}")
else:
    print("No checkpoint found, starting training from scratch.")
    last_epoch = 0

# Training loop - resume from the last epoch in checkpoint
EPOCHS = 5
for epoch in range(last_epoch, EPOCHS):
    model.train()  # set model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Save checkpoint after each epoch
    check_point = {
        "last_loss": avg_loss,
        "last_epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(check_point, f"./checkpoints/checkpoint_{epoch + 1}.pt")

print("Training complete. Checkpoints saved.")
