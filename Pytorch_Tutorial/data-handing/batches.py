import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import math

# data = np.loadtxt("data.csv")
# for epoch in range(1000):
#     x, y = data
#     # forward,
#     # backward,
#     # update weights

# # usually batches is made
# for epoch in range(1000):
#     for i in range(total_batches):
#         x_batch, y_batch = ...

# number of iterations = number of passes per epoch, where a pass is using batch_size


class WineDataSet(Dataset):
    """
    Assumed that label is the first column
    """

    def __init__(self, path_, y_at=0) -> None:
        super().__init__()
        # data loading
        xy = np.loadtxt(path_, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, y_at + 1 :])
        self.y = torch.from_numpy(xy[:, y_at])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


if __name__ == "__main__":
    dataset = WineDataSet("./wine.data")
    # first = dataset[0]
    # features, label = first
    # print(features)
    # print(label)

    batch_size = 4
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    # dataiter = iter(dataloader)
    # data = dataiter.next()
    # features, labels = data
    # print(features, labels)

    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / batch_size)
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # forward
            pass
            # backward
            pass
            # update
            pass
            if (i + 1) % 5 == 0:
                print(
                    f"epoch {epoch+1}/{num_epochs} : step {i+1}/{n_iterations}, inputs {inputs.shape}"
                )
