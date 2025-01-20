import torch

# Reshaping
o = torch.arange(12).reshape(4, 3)
print("o -", o)

# Viewing
r = o.view(3,4)
print("r -", r)
f = o.view(-1)
print("f -", f)

a, b = torch.arange(3).reshape(1, 3), torch.arange(3, 6).reshape(1, 3)
s = torch.stack((a, b), align=0)

# Squeezing

x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())

y = torch.squeeze(x)
print(y.size())

# Unsqueezing

x = torch.tensor([1, 2, 3, 4])
print(torch.unsqueeze(x, 0))

print(torch.unsqueeze(x, 1))

# Permute

x = torch.randn(2, 3, 5)
print(x.size())

print(torch.permute(x, (2, 0, 1)).size())