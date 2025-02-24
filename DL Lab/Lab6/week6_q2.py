import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained AlexNet Model & Modify Last Layer
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset Class
class CatsAndDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.class_map = {"dog": 0, "cat": 1}
        self.transform = transform
        for label in ["dogs", "cats"]:
            class_path = os.path.join(root_dir, label)
            for img_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.data.append((img_path, self.class_map[label[:-1]]))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Dataset & DataLoader
batch_size = 4
train_dataset = CatsAndDogsDataset("./cats_and_dogs_filtered/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CatsAndDogsDataset("./cats_and_dogs_filtered/validation", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the Model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

# Save Model
torch.save(model.state_dict(), "./ModelFiles/alexnet_cats_dogs.pth")
print("Model saved successfully!")

# Evaluate the Model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")