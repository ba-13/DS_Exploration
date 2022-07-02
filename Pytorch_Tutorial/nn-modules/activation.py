from turtle import forward
import torch
import torch.nn as nn


class NeuralNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNetwork1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


class NeuralNetwork2(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNetwork2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # using torch api instead of defining as different layer itself
        out = torch.relu(self.linear(x))
        out = torch.sigmoid(self.linear2(out))
        return out
