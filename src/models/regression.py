import torch


class Model(torch.nn.Module):
    def __init__(self, output_units, *input_dicts):
        super().__init__()
        self.output = torch.nn.Linear(27, output_units)

    def forward(self, *x):
        # x, *xs = x
        x = self.output(x[0])
        return x
