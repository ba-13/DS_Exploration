import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset


class WineDataSet(Dataset):
    def __init__(self, path, transform=None) -> None:
        super().__init__()
        xy = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 1:]
        self.y = xy[
            :, [0]
        ]  # note the difference between [:, 0] and [:, [0]], the later saves it still as an array

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


if __name__ == "__main__":
    dataset = WineDataSet(path="./wine.data", transform=ToTensor())
    first_data = dataset[0]
    print(first_data)
    features, label = first_data

    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
    dataset = WineDataSet(path="./wine.data", transform=composed)
    first_data = dataset[0]
    print(first_data)
