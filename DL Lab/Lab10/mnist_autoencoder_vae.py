import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs = 20
latent_dim = 10

# MNIST DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 1. AutoEncoder Implementation
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

# 2. Variational AutoEncoder Implementation
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, latent_dim)
        self.log_var = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        x = x.view(-1, 784)
        enc = self.encoder(x)
        mean = self.mean(enc)
        log_var = self.log_var(enc)
        z = self.reparameterize(mean, log_var)
        decoded = self.decoder(z)
        return decoded.view(-1, 1, 28, 28), mean, log_var

# Loss function for VAE
def vae_loss_function(recon_x, x, mean, log_var):
    x_rescaled = (x + 1) / 2
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x_rescaled.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD

# Training function
def train_model(model, optimizer, loss_fn, model_type='ae'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            if model_type == 'vae':
                recon_batch, mean, log_var = model(data)
                loss = loss_fn(recon_batch, data, mean, log_var)
            else:
                recon_batch = model(data)
                loss = loss_fn(recon_batch.view(-1, 784), data.view(-1, 784))
                
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_data_loader.dataset):.4f}')

# Generate digits function for VAE
def generate_digit_vae(model):
    model.eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        generated = model.decoder(z).detach().cpu().reshape(28, 28)
        plt.figure(figsize=(2, 2))
        plt.imshow(generated, cmap='gray')
        plt.axis('off')
        plt.savefig('generated_digit.png')
        plt.close()

# Initialize and train AutoEncoder
ae_model = AutoEncoder().to(device)
ae_optimizer = optim.Adam(ae_model.parameters(), lr=learning_rate)
ae_loss_fn = nn.MSELoss()
print("Training AutoEncoder...")
train_model(ae_model, ae_optimizer, ae_loss_fn, 'ae')

# Initialize and train Variational AutoEncoder
vae_model = VariationalAutoEncoder().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
print("\nTraining Variational AutoEncoder...")
train_model(vae_model, vae_optimizer, vae_loss_function, 'vae')

# Generate sample digits
print("Generating sample digits...")
generate_digit_vae(vae_model)
