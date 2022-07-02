import numpy as np
import torch.nn as nn
import torch
from softmax import softmax


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


actual_label = [0, 1, 2]
num_classes = 3
trained_raw_scores = [[2.0, 0.1, 1.0], [0.3, 4.0, 0.9], [0.1, 0.6, 1.0]]
random_raw_scores = [[0.05, 2.0, 0.3], [0.3, 0.3, 0.7], [0.5, 3, 3]]


def basicTorchExample():
    # torch's cross entropy already has softmax implemented, so pass raw scores (logits)
    # y should be class labels and not one hot encoded.

    loss = nn.CrossEntropyLoss()
    Y = torch.tensor([actual_label[0]])
    # num_samples * num_classes = 1x3
    Y_pred_good = torch.tensor([[2.0, 0.1, 1.0]])
    Y_pred_bad = torch.tensor([[0.05, 2.0, 0.3]])
    l1 = loss(Y_pred_good, Y)
    l2 = loss(Y_pred_bad, Y)
    print(f"Loss 1 torch : {l1.item():.4f}")
    print(f"Loss 2 torch : {l2.item():.4f}")

    _, predictions1 = torch.max(Y_pred_good, 1)
    _, predictions2 = torch.max(Y_pred_bad, 1)
    print("trained_raw_scores prediction :", predictions1.item())
    print("random_raw_scores prediction :", predictions2.item())


def advTorchExample():
    loss = nn.CrossEntropyLoss()
    Y = torch.tensor(actual_label)
    Y_pred_good = torch.tensor(trained_raw_scores)
    Y_pred_bad = torch.tensor(random_raw_scores)

    l1 = loss(Y_pred_good, Y)
    l2 = loss(Y_pred_bad, Y)
    print(f"Loss 1 torch : {l1.item():.4f}")
    print(f"Loss 2 torch : {l2.item():.4f}")
    _, predictions1 = torch.max(Y_pred_good, 1)
    _, predictions2 = torch.max(Y_pred_bad, 1)
    print("trained_raw_scores prediction :", predictions1)
    print("random_raw_scores prediction :", predictions2)


if __name__ == "__main__":

    Y = np.zeros(num_classes)
    Y[actual_label[0]] = 1
    # one hot encoded y as input

    Y_pred_good = np.array(softmax(trained_raw_scores))
    Y_pred_bad = np.array(softmax(random_raw_scores))
    # softmax fed y_pred as input

    l1 = cross_entropy(Y, Y_pred_good)
    l2 = cross_entropy(Y, Y_pred_bad)
    print(f"Loss 1 np : {l1:.4f}")
    print(f"Loss 2 np : {l2:.4f}")

    basicTorchExample()
    advTorchExample()
