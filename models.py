import torch
import torch.nn as nn
import copy

def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return
    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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

# A more flexbile version of 'MLP'
class MLP2(nn.Module):
    def __init__(self, input_size, n_hidden, n_units, out_size, out_sigmoid=True):
        super(MLP2, self).__init__()
        self.fc_in = nn.Linear(input_size, n_units)
        self.n_hidden = n_hidden
        self.fc_hiddens = clones(nn.Linear(n_units, n_units), n_hidden - 1)
        self.fc_out = nn.Linear(n_units, out_size)
        self.out_sigmoid = out_sigmoid

    def forward(self, x):
        out = self.fc_in(x)
        out = ReLU()(out)
        for i in range(self.n_hidden - 1):
            out = self.fc_hiddens[i](out)
            out = ReLU()(out)
        out = self.fc_out(out)

        if self.out_sigmoid:
            out = torch.sigmoid(out)
        return out