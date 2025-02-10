import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the CNN Model (FashionMNIST_CNN.py)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier()
model.load_state_dict(torch.load("./ModelFiles/model.pt"))
model.to(device)

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
fashion_mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(fashion_mnist_testset, batch_size=64, shuffle=False)

# Print model's state_dict. We are printing only the size of the parameter
print("Model's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t",model.state_dict()[param_tensor].size())
print()

# Evaluate the model
model.eval()  # set the model to evaluation mode
correct = 0
total = 0
for i, (inputs, labels) in enumerate(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)

    # Perform forward pass
    outputs = model(inputs)

    # Get predictions (the class with the highest score)
    _, predicted = torch.max(outputs, 1)

    # Calculate the accuracy
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    # Print predictions for inspection (Optional)
    # print(f"True labels: {labels}")
    # print(f"Predicted labels: {predicted}")

accuracy = 100.0 * correct / total
print(f"The overall accuracy is {accuracy}%")
