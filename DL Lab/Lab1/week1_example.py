import torch
# print(torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Count number of devices
print(torch.cuda.device_count())

# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])
# Tensor not on GPU
print(tensor, tensor.device)
# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

# If tensor is on GPU, can't transform it to NumPy (this will error)
# tensor_on_gpu.numpy()

# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)


