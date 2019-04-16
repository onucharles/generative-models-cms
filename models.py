import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = x.view(x.size(0), -1)
        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features=100, out_features=256, bias=True)
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), padding=4)
        #self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=2)
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=2)
        self.elu = nn.ELU(alpha=1.)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear(z)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.elu(z)
        z = self.conv0(z)
        z = self.elu(z)
        z = F.interpolate(z, scale_factor=2, mode='bilinear')
        z = self.conv1(z)
        z = self.elu(z)
        z = F.interpolate(z, scale_factor=2, mode='bilinear')
        z = self.conv2(z)
        z = self.elu(z)
        x_tilde_logits = self.conv3(z)
        #x_tilde = self.sigmoid(v)
        return x_tilde_logits


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def reparametrize(self, mean, logvar):
        epsilon = torch.randn_like(logvar)
        z = mean + torch.exp(logvar / 2) * epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparametrize(mean, logvar)
        x_tilde_logits = self.decoder(z)
        return x_tilde_logits, mean, logvar