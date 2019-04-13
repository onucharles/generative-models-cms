import numpy as np
import torch
import torch.nn as nn
from torch.utils import data


#Load Dataset
def load_data(train_path, test_path, val_path=None, batch_size=32):
    with open(train_path) as f:
        lines = f.readlines()
    x_train = np.array([[float(i) for i in line.split(' ')] for line in lines])
    y_train = np.zeros((x_train.shape[0], 1))
    train = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)

    with open(test_path) as f:
        lines = f.readlines()
    x_test = np.array([[float(i) for i in line.split(' ')] for line in lines])
    y_test = np.zeros((x_test.shape[0], 1))
    test = data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=True)


    if val_path is not None:
        with open(train_path) as f:
            lines = f.readlines()
        x_val = np.array([[float(i) for i in line.split(' ')] for line in lines])
        y_val= np.zeros((x_val.shape[0], 1))
        validation = data.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
        val_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=True)
        return (train_loader, val_loader, test_loader)
    else:
        return (train_loader, test_loader)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool0 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(7, 7), padding=0)
        self.linear_mean = nn.Linear(in_features=256, out_features=100, bias=True)
        self.linear_logvar= nn.Linear(in_features=256, out_features=100, bias=True)
        self.elu = nn.ELU(alpha=1.)

    def forward(self, x):
        x = self.conv0(x)
        x = self.elu(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.pool(1)
        x = self.conv2(x)
        x = self.elu(x)
        x = x.view(x.size(0), -1)
        mean = self.Linear_mean(x)
        logvar = self.linear_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_features=100, out_features=256, bias=True)
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), padding=4)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=2)
        self.elu = nn.ELU(alpha=1.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.elu(x)
        x = self.conv0(x)
        x = self.elu(x)
        x = self.upsample0(x)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mean, logvar = self.encoder(x)
        epsilon = torch.randn_like(x)
        z = mean + torch.exp(logvar/2) * epsilon
        x_hat = self.decoder(z)
        return x_hat


seed = 1111
np.random.seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    #load data
    train_path = "binarized_mnist_train.amat"
    val_path = "binarized_mnist_val.amat"
    test_path = "binarized_mnist_test.amat"
    train_loader, val_loader, test_loader = load_data(train_path, test_path, val_path, batch_size=32)



