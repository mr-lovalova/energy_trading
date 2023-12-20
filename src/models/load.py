import pickle


def data(path):
    with open(path + "train.pkl", "rb") as f:
        train_loader = pickle.load(f)

    with open(path + "valid.pkl", "rb") as f:
        valid_loader = pickle.load(f)

    with open(path + "test.pkl", "rb") as f:
        test_loader = pickle.load(f)

    return train_loader, valid_loader, test_loader
