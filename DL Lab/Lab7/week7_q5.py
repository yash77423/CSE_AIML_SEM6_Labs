import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

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

# Define Neural Network Architecture
class CatDogModel(nn.Module):
    def __init__(self):
        super(CatDogModel, self).__init__()
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

# Training with Early Stopping
def train_model_with_early_stopping(model, train_loader, val_loader, num_epochs=50, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_validation_loss = float('inf')
    current_patience = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

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

        # Validation step
        model.eval()  # Set model to evaluation mode
        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # No gradients required for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Average validation loss
        validation_loss /= len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {running_loss/len(train_loader):.4f}, '
              f'Validation Loss: {validation_loss:.4f}, '
              f'Validation Accuracy: {100 * correct / total:.2f}%')

        # Check for early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_patience = 0
            # Save the model with the best validation loss
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            current_patience += 1

        # If patience is exceeded, stop training
        if current_patience > patience:
            print(f"Early stopping triggered! No improvement in validation loss for {patience} epochs.")
            break

# Initialize model, train with early stopping
model_with_early_stopping = CatDogModel().to(device)
train_model_with_early_stopping(model_with_early_stopping, train_loader, val_loader, num_epochs=50, patience=5)

# Test model performance on the test set
def test_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Load the best model and test it
model_with_early_stopping.load_state_dict(torch.load('best_model.pth'))
test_model(model_with_early_stopping, test_loader)
