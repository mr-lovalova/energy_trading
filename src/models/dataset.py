import os
import torch
from torch.utils.data import Dataset


class ProductionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


path = "data/"
X = torch.load(os.path.join(path, "processed/production/X_wind.pt"))
y = torch.load(os.path.join(path, "processed/production/y_wind.pt"))

X = torch.Tensor(X)
y = torch.Tensor(y)
X.to(dtype=torch.float32)

print(X.dtype)
print(y.dtype)
dataset = ProductionDataset(X, y)
