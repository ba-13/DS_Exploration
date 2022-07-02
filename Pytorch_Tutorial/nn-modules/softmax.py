import numpy as np
import torch


def softmax(x):
    tmp = np.exp(x)
    return tmp / np.sum(tmp, axis=0)


if __name__ == "__main__":
    x = np.array([2.0, 1.0, 0.1])
    outputs = softmax(x)
    print("softmax numpy : ", outputs)

    x = torch.from_numpy(x)
    outputs = torch.softmax(x, dim=0)  # softmax along which axis.
    print("softmax torch : ", outputs)
