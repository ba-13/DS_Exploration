import torch
import torch.nn as nn


class NeuralNetBinary(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super(NeuralNetBinary, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


if __name__ == "__main__":
    model = NeuralNetBinary(input_size=28 * 28, hidden_size=5)
    criterion = nn.BCELoss()  # has sigmoid at the end
