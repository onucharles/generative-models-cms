import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            return torch.relu_(input)
        else:
            return torch.relu(input)

class MLP(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, out_size, out_sigmoid=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, out_size)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.out_sigmoid = out_sigmoid

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)

        if self.out_sigmoid:
            out = torch.sigmoid(out)
        return out