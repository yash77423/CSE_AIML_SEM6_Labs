import string
import torch
import torch.nn as nn

# Constants and Setup
letters = string.ascii_lowercase + '#' + ' ' # Character set including lowercase letters and EOF marker
n_letters = len(letters)               # Total number of unique characters (27)

# Character to Tensor Conversion Functions
def ltt(ch):    
    """
    Convert a single character to one-hot encoded tensor
    Args:
        ch: single character
    Returns:
        torch.tensor: one-hot encoded vector of size n_letters
    """
    ans = torch.zeros(n_letters) 
    ans[letters.find(ch)] = 1   
    return ans

def getLine(s):
    """
    Convert string to tensor of one-hot encoded characters
    Args:
        s: input string
    Returns:
        torch.tensor: 3D tensor of shape (len(s), 1, n_letters)
    """
    ans = [ltt(c) for c in s]
    return torch.cat(ans, dim=0).view(len(s), 1, n_letters)

# LSTM Model Definition
class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize LSTM model
        Args:
            input_dim: size of input (number of letters)
            hidden_dim: size of hidden state
        """
        super(MyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim)  # Single layer LSTM

    def forward(self, inp, hc):
        """
        Forward pass of LSTM
        Args:
            inp: input tensor of shape (seq_len, batch, input_dim)
            hc: tuple of (hidden state, cell state)
        Returns:
            torch.tensor: output tensor of shape (seq_len, batch, hidden_dim)
        """
        output, _ = self.LSTM(inp, hc)
        return output

# Data Preparation
data = "i love neural networks"  # Training data
EOF = "#"                         # End of sequence marker
data = data.lower()              # Convert to lowercase
seq_len = len(data)              # Length of input sequence

# Model Initialization
hidden_dim = n_letters           # Set hidden dimension same as input dimension
model = MyLSTM(n_letters, hidden_dim)  # Create LSTM model instance
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)  # Adam optimizer
LOSS = torch.nn.CrossEntropyLoss()  # Loss function for classification

# Target Preparation
targets = []
for x in data[1:] + EOF:
    target_idx = letters.find(x)
    if target_idx == -1:  # Debug check
        print(f"Character '{x}' not found in letters!")
    targets.append(target_idx)
targets = torch.tensor(targets)
print("Targets:", targets)  # Debug print to verify targets

# Input Preparation
inpl = [ltt(c) for c in data]    # Convert input string to list of tensors
inp = torch.cat(inpl, dim=0)     # Concatenate into single tensor
inp = inp.view(seq_len, 1, n_letters)  # Reshape to (seq_len, batch_size=1, n_letters)

# Training
n_iters = 150                    # Number of training iterations

for itr in range(n_iters):
    """Training loop"""
    model.zero_grad()            # Clear previous gradients
    h = torch.rand(hidden_dim).view(1, 1, hidden_dim)  # Random initial hidden state
    c = torch.rand(hidden_dim).view(1, 1, hidden_dim)  # Random initial cell state
    output = model(inp, (h, c))  # Forward pass
    output = output.view(seq_len, n_letters)  # Reshape for loss calculation
    loss = LOSS(output, targets)  # Calculate loss
    if itr % 10 == 0:            # Print loss every 10 iterations
        print(f"Iteration {itr}, Loss: {loss.item()}")
    loss.backward()              # Backpropagation
    optimizer.step()             # Update weights

# Prediction Function
def predict(s):
    """
    Predict next character given an input string
    Args:
        s: input string
    Returns:
        str: predicted next character
    """
    print(f"Input string: {s}")
    inp = getLine(s)            # Convert input to tensor
    print(f"\nInput tensor: {inp}")
    h = torch.rand(1, 1, hidden_dim)  # Random initial hidden state
    c = torch.rand(1, 1, hidden_dim)  # Random initial cell state
    out = model(inp, (h, c))    # Get model output
    # Get the predicted character from the last output
    predicted_char = letters[out[-1][0].topk(1)[1].detach().numpy().item()]
    print(f"Predicted next character: {predicted_char}")
    return predicted_char

# Test Prediction
test_str = "i love neu"  # Test string
predict(test_str)          # Make prediction