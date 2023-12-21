import torch
from torch import nn
import dataset
from loops import validation_loop, train_loop

# from ray import tune
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler

from model import Model

# from regression import Model
import matplotlib.pyplot as plt
import helpers

TEST_SPLIT = 0.05
VALID_SPLIT = 0.10

# HYPERPARAMETERS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 7000  # 6000

# MODEL
NAME = "solar_precip_temp"
CREATE_NEW_DATA = True
TYPE = "production"
YEAR = "2022_solar"
DATA_PATH = f"data/processed/{TYPE}/{YEAR}/"
MODEL_PATH = f"model/{NAME}/"
TARGET = "production"
FEATURES = ["radiation", "precip", "temperature"]

train_loader, valid_loader, test_loader = dataset.get(
    CREATE_NEW_DATA, DATA_PATH, MODEL_PATH, TARGET, *FEATURES
)

print(len(train_loader), len(valid_loader), len(test_loader))

num_features, num_target = helpers.get_num_input_output(train_loader)
print("NUM_TARGET", num_features)

input_dicts = []
for xs in num_features:
    input = {
        "NUM_FEATURES": xs,
        "OUTPUT_UNITS": 64,
        "DROPOUT": 0.1,
    }
    input_dicts.append(input)


model = Model(num_target, *input_dicts)

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
# plt.show()
