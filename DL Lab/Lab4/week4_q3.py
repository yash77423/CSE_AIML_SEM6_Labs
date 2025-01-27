import torch
import torch.nn as nn
import numpy as np

# Create a model instance and load it to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.linear1 = nn.Linear(2, 2, bias=True)  # 2 inputs, 2 hidden neurons
        self.activation1 = nn.ReLU()  # ReLU activation for hidden layer
        self.linear2 = nn.Linear(2, 1, bias=True)  # 2 hidden neurons, 1 output
        self.activation2 = nn.Sigmoid()  # Sigmoid activation for output layer

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)  # Apply ReLU activation
        x = self.linear2(x)
        x = self.activation2(x)  # Apply Sigmoid activation for output
        return x


# Manually define the input (X) for XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)

# Initialize the model and load it to device (assuming the model is trained)
model = XORModel().to(device)

# Extract system-generated weights and biases for Linear1 and Linear2
w1, b1 = model.linear1.weight.data, model.linear1.bias.data
w2, b2 = model.linear2.weight.data, model.linear2.bias.data

# Print the extracted values of weights and biases
print("Linear1 Weights (W1):", w1)
print("Linear1 Biases (b1):", b1)
print("Linear2 Weights (W2):", w2)
print("Linear2 Biases (b2):", b2)

# Step 1: Compute Linear1 output: out1 = X * W1 + b1
out1 = torch.matmul(X, w1.T) + b1  # Linear transformation
print("Output after Linear1:", out1)

# Step 2: Apply ReLU activation: ReLU(out1) = max(0, out1)
relu_out1 = torch.relu(out1)
print("Output after ReLU activation:", relu_out1)

# Step 3: Compute Linear2 output: out2 = ReLU(out1) * W2 + b2
out2 = torch.matmul(relu_out1, w2.T) + b2  # Linear transformation
print("Output after Linear2:", out2)

# Step 4: Apply Sigmoid activation: Sigmoid(out2) = 1 / (1 + exp(-out2))
sigmoid_out = torch.sigmoid(out2)
print("Final Output after Sigmoid activation:", sigmoid_out)
