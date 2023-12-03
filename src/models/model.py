import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(55, 200)
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(200, 200)
        self.output = torch.nn.Linear(200, 99)

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.output(x)
        return x


model = TinyModel()

print("The model:")
print(model)
