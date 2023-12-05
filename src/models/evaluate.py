import torch
import pickle
from matplotlib import pyplot as plt
import numpy as np
from model import Model
import helpers
from loops import test_loop
from municipalities import MUNICIPALITIES

# MODEL
NAME = "wind_pressure"
MODEL_PATH = f"model/{NAME}/"


model_dict = torch.load(MODEL_PATH + "model.pt")

with open(MODEL_PATH + "test.pkl", "rb") as f:
    test_loader = pickle.load(f)

num_features, num_target = helpers.get_num_input_output(test_loader)
model = Model(num_features)

model.load_state_dict(model_dict)

loss_fn = torch.nn.MSELoss(reduction="none")
avg_test_loss, test_loss_values = test_loop(test_loader, model, loss_fn)


*X, y = next(iter(test_loader))
model.eval()
pred = model(*X)

example_idx = 0
single_example_pred = pred[example_idx]
single_example_ground_truth = y[example_idx]

# Plotting
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
plt.title("Comparison between prediction and ground truth")
plt.legend()
plt.savefig(MODEL_PATH + "avg_err.png")

with open(MODEL_PATH + "info.txt", "w") as f:
    f.write(f"Average test loss: {avg_test_loss} MWh")
