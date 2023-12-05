import torch

# *input_dicts


class Model(torch.nn.Module):
    # def __init__(self, *num_features, out1=64, out2=64):
    def __init__(self, *input_dicts):
        super().__init__()
        self.sequantial_inputs = []
        first_input_units = 0
        for input in input_dicts:
            first_input_units += input["OUTPUT_UNITS"]
            seq = torch.nn.Sequential(
                torch.nn.Linear(input["NUM_FEATURES"], input["OUTPUT_UNITS"]),
                torch.nn.Dropout(input["DROPOUT"]),
            )
            self.sequantial_inputs.append(seq)

        self.linear1 = torch.nn.Linear(first_input_units, 1024)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.output = torch.nn.Linear(1024, 99)

    def forward(self, *xs):
        inputs = [self.sequantial_inputs[count](x) for count, x in enumerate(xs)]
        x = torch.cat(inputs, dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.output(x)
        return x
