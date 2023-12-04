import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset import dataset
from loops import test_loop, train_loop
from model import model
import matplotlib.pyplot as plt

TEST_SPLIT = 0.025
VALID_SPLIT = 0.5

# HYPERPARAMETERS
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 4000


test_size = int(len(dataset) * TEST_SPLIT)
valid_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - test_size - valid_size
train_data, valid_data, test_data = random_split(
    dataset, [train_size, valid_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

test_loss_values = []
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    avg_test_loss, test_loss_epoch = test_loop(test_loader, model, loss_fn)
    test_loss_values.extend(test_loss_epoch)
print("Done!")

# Plotting the Mean Squared Error values
plt.plot(test_loss_values, label="Test Loss")
plt.xlabel("Batch")
plt.ylabel("MWh Error")
plt.legend()
plt.show()


X, y = next(iter(test_loader))
model.eval()
pred = model(X)
print("PREDICTION")
print(pred)
print("Y")
print(y)

import numpy as np

example_idx = 0
single_example_pred = pred[example_idx]
single_example_ground_truth = y[example_idx]

# Plotting
plt.figure(figsize=(10, 6))

# Plotting ground truth bars
plt.bar(
    np.arange(len(single_example_ground_truth)),
    single_example_ground_truth,
    label="Ground Truth",
)

# Plotting prediction markers
plt.plot(
    np.arange(len(single_example_pred)),
    single_example_pred.detach().numpy(),
    "rx",
    markersize=8,
    label="Prediction",
)

plt.xlabel("Output Dimension")
plt.ylabel("Values")
plt.title("Comparison between Prediction and Ground Truth for a Single Example")
plt.legend()
plt.show()
