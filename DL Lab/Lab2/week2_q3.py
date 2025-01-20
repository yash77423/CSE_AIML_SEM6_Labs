import torch
import numpy as np

def grad_sigmoid_manual(x):
    """ Implements the gradient of the logistic sigmoid function
        sigma(x) = 1 / (1 + e^{-x})
    """
    # Forward pass, keeping track of intermediate values for use in the backward pass
    a = -x  # -x in denominator
    b = np.exp(a)  # e^{-x} in denominator
    c = 1 + b  # 1 + e^{-x} in denominator
    s = 1.0 / c  # Final result, 1.0 / (1 + e^{-x})

    # Backward pass
    dsdc = (-1.0 / (c ** 2))
    dsdb = dsdc * 1
    dsda = dsdb * np.exp(a)
    dsdx = dsda * (-1)

    return dsdx


def sigmoid(x):
    y = 1.0 / (1.0 + torch.exp(-x))
    return y


input_x = 2.0
x = torch.tensor(input_x).requires_grad_(True)
y = sigmoid(x)
y.backward()
# Compare the results of manual and automatic gradient functions:
print('autograd:', x.grad.item())
print('manual:', grad_sigmoid_manual(input_x))
