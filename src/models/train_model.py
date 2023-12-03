import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset import dataset
from loops import test_loop, train_loop
from model import model
import matplotlib.pyplot as plt

TEST_SPLIT = 0.1
VALID_SPLIT = 0.15

# HYPERPARAMETERS
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10000


test_size = int(len(dataset) * TEST_SPLIT)
valid_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - test_size - valid_size
train_data, valid_data, test_data = random_split(
    dataset, [train_size, valid_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

print(valid_loader)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

test_loss_values = []
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    # test_loop(test_loader, model, loss_fn)
    avg_test_loss, test_loss_epoch = test_loop(test_loader, model, loss_fn)
    test_loss_values.extend(test_loss_epoch)
print("Done!")

# Plotting the Mean Squared Error values
plt.plot(test_loss_values, label="Test Loss")
plt.xlabel("Batch")
plt.ylabel("MWh Error")
plt.legend()
plt.show()
