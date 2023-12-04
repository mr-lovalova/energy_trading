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
X_hour = torch.load(os.path.join(path, "processed/production/X_hour.pt"))
X_month = torch.load(os.path.join(path, "processed/production/X_month.pt"))
y = torch.load(os.path.join(path, "processed/production/y_wind.pt"))

X = torch.Tensor(X)
X.to(dtype=torch.float32)
X_hour = torch.Tensor(X_hour)
X_hour = X_hour.to(dtype=torch.long)
X_month = torch.Tensor(X_month)
X_month = X_month.to(dtype=torch.long)
y = torch.Tensor(y)

print(X_month.dtype)

dataset = ProductionDataset(X, y)

# Check the shape of X_hour and X_month
print("Shape of X_hour:", X_hour.shape)
print("Shape of X_month:", X_month.shape)

# Check unique values in X_hour and X_month
unique_values_hour = torch.unique(X_hour)
unique_values_month = torch.unique(X_month)

print("Unique values in X_hour:", unique_values_hour)
print("Unique values in X_month:", unique_values_month)
