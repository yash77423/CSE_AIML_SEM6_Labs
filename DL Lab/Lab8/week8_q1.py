# Implement an RNN that predicts the next no in a Fibonacci sequence based on previous 3 nos.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Step 1: Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Step 2: RNN Model Class
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])  # Predict from last time step
        return out

# Step 3: General Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs, device='cpu'):
    model.to(device)
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    plt.plot(range(1, num_epochs+1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.show()

# Step 4: General Evaluation Loop
def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Loss: {avg_loss:.4f}')

# Step 5: Main Execution
# Corrected Data: Fibonacci sequences (3 previous numbers -> next number)
inputs = [[[0], [1], [1]],  # Predict 2
          [[1], [1], [2]],  # Predict 3
          [[1], [2], [3]],  # Predict 5
          [[2], [3], [5]]]  # Predict 8
targets = [[2], [3], [5], [8]]  # Shape: (4, 1)
inputs = torch.tensor(inputs, dtype=torch.float32)  # Shape: (4, 3, 1)
targets = torch.tensor(targets, dtype=torch.float32)

# Dataset and DataLoader
dataset = CustomDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(dataset, batch_size=2)  # Using same data for simplicity

# Model, Loss, Optimizer
model = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Calculate Trainable Parameters
total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print("Total number of trainable parameters =", total_params)

# Train and Evaluate
train_model(model, train_loader, criterion, optimizer, num_epochs=100)
evaluate_model(model, test_loader, criterion)

# Test a prediction
test_input = torch.tensor([[[2], [3], [5]]], dtype=torch.float32)  # Predict 8
with torch.no_grad():
    pred = model(test_input)
    print(f'Predicted next Fibonacci number: {pred.item():.2f}')  # Should be ~8
