import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class for Cat-Dog Classification
class CatDogDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        for label, class_name in enumerate(['cats', 'dogs']):
            class_path = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.endswith(".jpg") or img_name.endswith(".png"):
                    self.img_labels.append((os.path.join(class_path, img_name), label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for the dataset (resize and normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize for faster training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Load dataset
train_data = CatDogDataset(img_dir='./data/cats_and_dogs_filtered/train', transform=transform)
test_data = CatDogDataset(img_dir='./data/cats_and_dogs_filtered/validation', transform=transform)

# Split train and validation
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Dataloader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Custom Dropout Layer using Bernoulli Distribution
class CustomDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CustomDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x):
        if self.training:  # Only apply dropout during training
            dropout_mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout_prob)).to(x.device)
            return x * dropout_mask / (1 - self.dropout_prob)  # Scale the output during training
        return x  # During inference, just return the input

# Define Neural Network Architecture with Dropout
class CatDogModelWithDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CatDogModelWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)  # Adjusted for input size (128x128)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes: cat and dog
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with specified probability

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Define Neural Network Architecture with Custom Dropout
class CatDogModelWithCustomDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(CatDogModelWithCustomDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)
        self.custom_dropout = CustomDropout(dropout_prob)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.custom_dropout(x)  # Apply custom dropout
        x = self.fc2(x)
        return x

# Define Neural Network Architecture without Dropout (for comparison)
class CatDogModelNoDropout(nn.Module):
    def __init__(self):
        super(CatDogModelNoDropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 512)  # Adjusted for input size (128x128)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes: cat and dog

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the model with and without dropout
def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on validation set
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

# Initialize and train with dropout
model_with_dropout = CatDogModelWithDropout(dropout_prob=0.5).to(device)
print("Training model with Dropout:")
train_model(model_with_dropout, train_loader, val_loader, epochs=10)

# Initialize and train without dropout
model_no_dropout = CatDogModelNoDropout().to(device)
print("\nTraining model without Dropout:")
train_model(model_no_dropout, train_loader, val_loader, epochs=10)

# Initialize and train with custom dropout
model_with_custom_dropout = CatDogModelWithCustomDropout(dropout_prob=0.5).to(device)
print("\nTraining model with Custom Dropout:")
train_model(model_with_custom_dropout, train_loader, val_loader, epochs=10)
