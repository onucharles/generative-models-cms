import torch
import torch.nn as nn
import torch.nn.functional as F
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


# A VAE adjusted adjusted to 1x28x28 images
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


##Another VAE more adjusted to 3x32x32 images
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0)
        self.pool0 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(6, 6), padding=0)
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

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(in_features=100, out_features=256, bias=True)
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(5, 5), padding=4)
        #self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=3)
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), padding=2)
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
        x_tilde = self.conv3(z)
        return x_tilde


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder2()
        self.decoder = Decoder2()

    def reparametrize(self, mean, logvar):
        epsilon = torch.randn_like(logvar)
        z = mean + torch.exp(logvar / 2) * epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparametrize(mean, logvar)
        x_tilde_logits = self.decoder(z)
        return x_tilde_logits, mean, logvar


#ELBO using Binary Cross Entropy reconstruction loss (bernoulli p(x|z))
class Elbo_BCE(nn.Module):
    def __init__(self):
        super(Elbo_BCE, self).__init__()

    def forward(self, x, x_tilde_logits, mean, logvar):
        #std = torch.exp(logvar/2)
        #log_encoder = -0.5 * torch.sum(((z - mean)/std)**2,-1) - 0.5 * torch.sum(torch.log(2*np.pi*std**2),-1)
        #log_prior = -0.5 * torch.sum(z ** 2, -1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)

        x = x.reshape((x.shape[0],-1))
        x_tilde_logits = x_tilde_logits.reshape((x_tilde_logits.shape[0], -1))

        # Analytical KL divergence for a standard normal distribution prior and a
        # multivariate normal distribution with diagonal covariance q(z|x)
        kl = 0.5 * (-logvar - 1 + mean**2 + torch.exp(logvar)).sum(dim=-1)
        log_decoder = -torch.sum(F.binary_cross_entropy_with_logits(input=x_tilde_logits, target=x, reduction="none"), dim=-1)
        #log_decoder = (x * torch.log(x_tilde) + (1 - x) * torch.log(1 - x_tilde)).sum(dim=-1) #log likelihood (- cross entropy)
        loss = -(log_decoder - kl).mean()
        return loss

#ELBO using p(x|z) as normal distribution with variance of 1
class Elbo_Normal(nn.Module):
    def __init__(self):
        super(Elbo_CE, self).__init__()

    def forward(self, x, x_tilde_logits, mean, logvar):
        #std = torch.exp(logvar/2)
        #log_encoder = -0.5 * torch.sum(((z - mean)/std)**2,-1) - 0.5 * torch.sum(torch.log(2*np.pi*std**2),-1)
        #log_prior = -0.5 * torch.sum(z ** 2, -1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)

        x = x.reshape((x.shape[0],-1))
        x_tilde_logits = x_tilde_logits.reshape((x_tilde_logits.shape[0], -1))

        # Analytical KL divergence for a standard normal distribution prior and a
        # multivariate normal distribution with diagonal covariance q(z|x)
        kl = 0.5 * (-logvar - 1 + mean**2 + torch.exp(logvar)).sum(dim=-1)
        log_decoder = -torch.sum(F.binary_cross_entropy_with_logits(input=x_tilde_logits, target=x, reduction="none"), dim=-1)
        #log_decoder = (x * torch.log(x_tilde) + (1 - x) * torch.log(1 - x_tilde)).sum(dim=-1) #log likelihood (- cross entropy)
        loss = -(log_decoder - kl).mean()
        return loss