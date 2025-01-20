import torch
from matplotlib import pyplot as plt
from torch import no_grad

# Training examples in the dataset for the linear regression
x = torch.tensor(
    [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2,
     17.4, 19.5, 19.7, 21.2])
y = torch.tensor(
    [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8,
     15.2, 17.0, 17.2, 18.6])


# The parameters to be learnt: w, and b in the prediction: y_p = wx + b
b = torch.rand([1], requires_grad=True)
w = torch.rand([1], requires_grad=True)
print("The parameters are: {}, and {}".format(w, b))

# The learning rate is to set alpha = 0.001
learning_rate = torch.tensor(0.001)

# The list of loss values
loss_list = []

# Run the training loop for N epochs
for epochs in range(100):
    # Compute the average loss for the training samples
    loss = 0.0

    # Accumulate the loss for all samples
    for j in range(len(x)):
        y_p = w * x[j] + b
        loss += (y_p - y[j]) ** 2

    # Find the average loss
    loss = loss / len(x)

    # Add the loss to a list for plotting purpose
    loss_list.append(loss.item())

    # Compute the gradients using backward: dL/dw and dL/db
    loss.backward()

    # Without modifying the gradient in this block, perform the operation:
    with torch.no_grad():
        # Update the weight based on gradient descent
        # equivalently one may write w1.copy_(w1 - learning_rate * w1.grad)
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Reset the gradients for next epoch
    w.grad.zero_()
    b.grad.zero_()
    # w.grad = None
    # b.grad = None
    # prev_loss = loss

    # Display the parameters and loss
    print("The parameters are w = {}, b = {}, loss = {}".format(w, b, loss.item()))

# Display the plot
plt.plot(loss_list)
plt.show()