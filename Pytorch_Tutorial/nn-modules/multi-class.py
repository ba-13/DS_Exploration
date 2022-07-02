import torch
import torch.nn as nn


class NeuralNetMulti(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetMulti, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out  # no softmax needed at end


if __name__ == "__main__":
    model = NeuralNetMulti(input_size=28 * 28, hidden_size=5, num_classes=3)
    criterion = nn.CrossEntropyLoss()
