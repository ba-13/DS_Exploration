# 1) Design model (input_size, output_size, forward_pass)
# 2) Construct loss and optimizer
# 3) Training Step
# - forward pass : compute prediction and loss
# - backward pass : gradient computation
# - update weights

from cgi import test
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare Dataset
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape
print(f"n_samples : {n_samples}, n_features : {n_features}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# scale features, 0 mean and unit variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
# Model
# f = sigmoid(wx + b)
class LogisticRegress(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegress, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegress(n_features)

# Loss and Optimizer
learning_rate = 0.05
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
num_epochs = 1000
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % (num_epochs // 10) == 0:
        print(f"epoch {epoch+1} : loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_classes = y_predicted.round()
    acc = y_predicted_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f"\nAccuracy = {acc:.4f}")
