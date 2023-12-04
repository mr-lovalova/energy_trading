import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(169, 300)
        self.activation1 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(300, 300)
        self.activation2 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(300, 300)

        self.output = torch.nn.Linear(300, 99)

        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation2(x)

        x = self.output(x)
        return x


model = TinyModel()

print("The model:")
print(model)
