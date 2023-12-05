import torch
from torch import nn
import dataset
from loops import validation_loop, train_loop
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import helpers

TEST_SPLIT = 0.05
VALID_SPLIT = 0.05

# HYPERPARAMETERS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 8000

# MODEL
NAME = "wind_pressure"
TYPE = "production"
YEAR = "2022"
DATA_PATH = f"data/processed/{TYPE}/{YEAR}/"
MODEL_PATH = f"model/{NAME}/"
TARGET = "production"
FEATURES = ["pressure", "windspeed"]

train_loader, valid_loader, test_loader = dataset.get(
    True, DATA_PATH, MODEL_PATH, TARGET, *FEATURES
)


num_features, num_target = helpers.get_num_input_output(test_loader)
model = Model(num_features)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

train_loss = []
test_loss = []
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    avg_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
    avg_validation_loss = validation_loop(test_loader, model, loss_fn)
    test_loss.append(avg_validation_loss)
    train_loss.append(avg_train_loss)
print("Done!")

torch.save(model.state_dict(), MODEL_PATH + "model.pt")

# Plotting the Mean Squared Error values
train_loss = [x.detach().numpy() for x in train_loss]
print(len(test_loss))
plt.plot(test_loss, label="Validation Loss")
plt.plot(train_loss, label="train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(MODEL_PATH + "loss.png")
plt.show()
