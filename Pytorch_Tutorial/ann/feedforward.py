# MNIST
# Dataloader, Transformation
# Multilayer NN, Activation function
# Loss and Optimizer
# Training Loop via Batches
# Model Evaluation
# GPU Support
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameters
input_size = 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
print("Preparing data...")
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=False
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)
print("...Done.")

# * Testing out dataloader
# examples = iter(train_loader)
# samples, labels = examples.next()
# print(samples.shape, labels.shape)

# * Testing out samples graphically
# plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(samples[i][0], cmap="gray")
# plt.show()


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)


def train():
    print("Training...")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} - Step {i+1}/{n_total_steps} : Loss = {loss.item():.4f}"
                )
    print("...Done.")


train()

#! Testing
print("Testing now...")
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        labels = labels.cpu()
        outputs = outputs.cpu()
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]

        correct = (predictions == labels).numpy()
        num_false = len(np.where(correct == 0))
        n_correct += correct.sum()
        # print(correct.sum(), " out of ", 100)
    acc = 100.0 * n_correct / n_samples
    print(f"accuracy = {acc}")
