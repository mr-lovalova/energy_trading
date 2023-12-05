import torch


class Model(torch.nn.Module):
    def __init__(self, *num_features, out1=64, out2=64):
        super().__init__()
        # self.inputs = [torch.nn.Linear(num, 64) for num in num_features]
        self.input1 = torch.nn.Linear(50, out1)
        self.input2 = torch.nn.Linear(55, out2)
        self.activation1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(out1 + out2, 1024)
        self.activation2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.output = torch.nn.Linear(1024, 99)

        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, *xs):
        # xs = [self.inputs[x] for x in xs]
        x1, x2 = xs
        x1 = self.input1(x1)
        x2 = self.input2(x2)
        x = torch.cat((x1, x2), dim=1)

        # x = self.input(x)
        # x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation2(x)

        x = self.output(x)
        return x
