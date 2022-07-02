# 1) Design model (input_size, output_size, forward_pass)
# 2) Construct loss and optimizer
# 3) Training Step
# - forward pass : compute prediction and loss
# - backward pass : gradient computation
# - update weights

from pickletools import optimize
import torch.nn as nn
import torch
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Step 0 : Prepare Dataset
X_np, Y_np = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=1
)
X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(Y_np.astype(np.float32))
y = y.view(y.shape[0], 1)  # reshapes to samples*features shape

n_samples, n_features = X.shape

# Step 1 : Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Step 2 : Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Step 3 : Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()
    # this summing up the gradients into the .grad attribute

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1} : loss = {loss.item():.4f}")

# plot
predicted = model(X).detach().numpy()
plt.plot(X_np, Y_np, "ro")
plt.plot(X_np, predicted, "b")
plt.show()
