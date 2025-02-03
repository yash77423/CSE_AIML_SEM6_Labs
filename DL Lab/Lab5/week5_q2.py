import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

# Define an input image of shape (1, 1, 6, 6)
image = torch.rand(1, 1, 6, 6)

# Create a Conv2d layer with a random kernel of size (3, 3) and out_channels = 3
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False)

# Manually set the kernel of the Conv2d layer to match the kernel for functional conv2d
kernel = torch.rand(3, 1, 3, 3)  # (out_channels, in_channels, height, width)
conv_layer.weight.data = kernel  # Set the weights

# Apply convolution using nn.Conv2d
output_conv2d = conv_layer(image)

# Apply convolution using torch.nn.functional.conv2d
output_conv2d_func = F.conv2d(image, kernel)

print("Output from nn.Conv2d:")
print(output_conv2d)

print("\nOutput from F.conv2d:")
print(output_conv2d_func)
