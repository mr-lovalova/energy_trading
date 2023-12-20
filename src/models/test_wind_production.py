import torch
import pickle
from matplotlib import pyplot as plt

# import numpy
from model import Model

# from regression import Model
import numpy as np
import helpers
from loops import test_loop
from municipalities import MUNICIPALITIES
from torchinfo import summary


# MODEL
NAME = "wind_pressurenew"
MODEL_PATH = f"models/{NAME}/"


model_dict = torch.load(MODEL_PATH + "model.pt")

with open(MODEL_PATH + "test.pkl", "rb") as f:
    test_loader = pickle.load(f)

num_features, num_target = helpers.get_num_input_output(test_loader)

# Adjust the input_size according to your model's input
input_dicts = []
for xs in num_features:
    input = {
        "NUM_FEATURES": xs,
        "OUTPUT_UNITS": 32,
        "DROPOUT": 0.0,
    }
    input_dicts.append(input)

model = Model(num_target, *input_dicts)
model.load_state_dict(model_dict)

BATCH_SIZE = 128

input_size = [(BATCH_SIZE, xs) for xs in num_features]
model_summary = summary(model, input_size=input_size)

loss_fn = torch.nn.MSELoss(reduction="none")
avg_test_loss, test_loss_values = test_loop(test_loader, model, loss_fn)


*X, y = next(iter(test_loader))
model.eval()
pred = model(*X)

example_idx = 1
single_example_pred = pred[example_idx]
single_example_ground_truth = y[example_idx]


plt.figure(figsize=(24, 6))

# Plotting ground truth bars
plt.plot(
    np.arange(len(single_example_ground_truth)),
    single_example_ground_truth,
    label="Ground Truth single example",
    color="orange",
)

plt.xticks(np.arange(len(single_example_ground_truth)), MUNICIPALITIES, rotation=90)
# Plotting prediction markers
plt.plot(
    np.arange(len(single_example_pred)),
    single_example_pred.detach().numpy(),
    "rx",
    label="Prediction single example",
)


plt.bar(
    np.arange(len(single_example_pred)),
    torch.mean(y, dim=0).detach().numpy(),
    label="Avg MWh production per municipality",
)

plt.bar(
    np.arange(len(single_example_pred)),
    test_loss_values[0],
    label="Avg MWh error per municipality",
    color="purple",
)


plt.xlabel("Output Dimension")
plt.ylabel("Mwh")
plt.title(f"Comparison between prediction and ground truth for {NAME} model")
plt.legend()
plt.show()
