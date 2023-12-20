import torch

# *input_dicts


class Model(torch.nn.Module):
    def __init__(self, output_units, *input_dicts):
        super().__init__()
        input_layers = []
        first_input_units = 0
        for input in input_dicts:
            first_input_units += input["OUTPUT_UNITS"]
            seq = torch.nn.Sequential(
                torch.nn.Linear(input["NUM_FEATURES"], input["OUTPUT_UNITS"]),
                torch.nn.Dropout(input["DROPOUT"]),
            )
            input_layers.append(seq)

        self.input_layers = torch.nn.ModuleList(input_layers)
        self.linear1 = torch.nn.Linear(first_input_units, 1024)
        # self.linear1 = torch.nn.Linear(first_input_units, 1024)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.output = torch.nn.Linear(1024, output_units)

    def forward(self, *xs):
        inputs = [self.input_layers[count](x) for count, x in enumerate(xs)]
        x = torch.cat(inputs, dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.output(x)
        return x


"""from torchinfo import summary

summary(
    model, input_size=[(32, 50), (32, 55)]
)  # Adjust the input_size according to your model's input
"""
