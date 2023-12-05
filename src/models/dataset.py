import os
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


TEST_SPLIT = 0.025
VALID_SPLIT = 0.05

# HYPERPARAMETERS
BATCH_SIZE = 128

#
YEAR = "2022"
TYPE = "production"

# X = torch.load(os.path.join(path, "processed/production/X_wind.pt"))
# = torch.load(os.path.join(path, "processed/production/y_wind.pt"))
# X = torch.load(os.path.join(PATH, "processed/production/X_2022winddir.pt"))
# y = torch.load(os.path.join(PATH, "processed/production/y_2022winddir.pt"))
# X = torch.load(os.path.join(path, "processed/production/X_2022temp.pt"))
# y = torch.load(os.path.join(path, "processed/production/y_2022temp.pt"))
# X = torch.load(os.path.join(path, "processed/production/X_2022temppreciprad.pt"))
# y = torch.load(os.path.join(path, "processed/production/y_2022temppreciprad.pt"))


class ProductionDataset(Dataset):
    def __init__(self, Xs, y):
        self.Xs = Xs
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        out = [X[index] for X in self.Xs]
        out.append(self.y[index])
        return out


# X = torch.Tensor(X)
# X.to(dtype=torch.float32)
# y = torch.Tensor(y)

# dataset = ProductionDataset(X, y)


def get_features(PATH, target, *features):
    y = torch.load(PATH + target + ".pt")
    Xs = [torch.Tensor(torch.load(PATH + feature + ".pt")) for feature in features]
    return Xs, y


def split_data(dataset, TEST_SPLIT, VALID_SPLIT):
    test_size = int(len(dataset) * TEST_SPLIT)
    valid_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - test_size - valid_size
    train_data, valid_data, test_data = random_split(
        dataset, [train_size, valid_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, valid_loader, test_loader


def generate(PATH, TENSOR_PATH, target, *features):
    Xs, y = get_features(PATH, target, *features)

    dataset = ProductionDataset(Xs, y)

    train, valid, test = split_data(dataset, TEST_SPLIT, VALID_SPLIT)
    *X, y = next(iter(train))

    with open(TENSOR_PATH + "train.pkl", "wb") as f:
        pickle.dump(train, f)

    with open(TENSOR_PATH + "valid.pkl", "wb") as f:
        pickle.dump(valid, f)

    with open(TENSOR_PATH + "test.pkl", "wb") as f:
        pickle.dump(test, f)

    return train, valid, test


def load(path):
    with open(path + "train.pkl", "rb") as f:
        train_loader = pickle.load(f)

    with open(path + "valid.pkl", "rb") as f:
        valid_loader = pickle.load(f)

    with open(path + "test.pkl", "rb") as f:
        test_loader = pickle.load(f)

    return train_loader, valid_loader, test_loader


def get(create, DATA_PATH, MODEL_PATH, target, *features):
    if create:
        return generate(DATA_PATH, MODEL_PATH, target, *features)
    return load(MODEL_PATH)


# PATH = f"data/processed/{TYPE}/{YEAR}/"
# generate(PATH, "TEST", "production", "pressure", "windspeed")
# a, b, c = get(True, TYPE, YEAR, "TEST2", "production", "pressure", "windspeed")
