import torch
import torch.nn.functional as F

# Define the input image (6x6)
image = torch.rand(6, 6)
print("image=\n", image)

# Add batch and channel dimensions (1, 1, 6, 6)
image = image.unsqueeze(dim=0).unsqueeze(dim=0)
print("image.shape=", image.shape)

# Define the kernel (3x3)
kernel = torch.ones(3, 3)
print("kernel=\n", kernel)

# Add batch and channel dimensions (1, 1, 3, 3)
kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)

# Perform convolution with stride=1 and padding=0
outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("outimage=\n", outimage)
print("outimage.shape=", outimage.shape)

# Example 1: Stride = 1, Padding = 0
outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("Output shape (stride=1, padding=0):", outimage.shape)

# Example 2: Stride = 2, Padding = 0
outimage = F.conv2d(image, kernel, stride=2, padding=0)
print("Output shape (stride=2, padding=0):", outimage.shape)

# Example 3: Stride = 1, Padding = 1
outimage = F.conv2d(image, kernel, stride=1, padding=1)
print("Output shape (stride=1, padding=1):", outimage.shape)

def count_parameters(kernel):
    return kernel.numel()

print("Number of parameters in the kernel:", count_parameters(kernel))