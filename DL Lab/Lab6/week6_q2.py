import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Define image pre-processing transformations
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the smaller side to 256px
    transforms.CenterCrop(224),  # Crop to 224x224 to match AlexNet's input size
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet's statistics
])

# Step 2: Load and prepare the dataset
train_dir = './cats_and_dogs_filtered/train'  # Directory for the training images
valid_dir = './cats_and_dogs_filtered/validation'  # Directory for validation images

# Load the training and validation datasets
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
valid_data = datasets.ImageFolder(root=valid_dir, transform=transform)

# Create DataLoader for batching and shuffling
batch_size = 4  # You can adjust this based on available GPU memory
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

# Step 3: Set up the pre-trained AlexNet model
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

# Modify the classifier to have 2 output classes (cats and dogs)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 4: Define Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # SGD optimizer

# Step 5: Define the training loop
epochs = 5  # You can adjust the number of epochs
train_loss = []
valid_loss = []

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    # Calculate training accuracy
    train_acc = 100 * correct_preds / total_preds
    train_loss.append(running_loss / len(train_loader))

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')

    # Validate the model
    model.eval()  # Set the model to evaluation mode
    valid_running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            valid_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    # Calculate validation accuracy
    valid_acc = 100 * correct_preds / total_preds
    valid_loss.append(valid_running_loss / len(valid_loader))
    print(f'Validation Loss: {valid_running_loss/len(valid_loader):.4f}, Accuracy: {valid_acc:.2f}%')

print("Training complete")

# Step 6: Save the model after training
torch.save(model.state_dict(), './ModelFiles/alexnet_finetuned.pth')
print("Model saved as 'alexnet_finetuned.pth'")

