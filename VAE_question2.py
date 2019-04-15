import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import time


#Load Dataset
def load_data(train_path, test_path, val_path=None, batch_size=32):
    with open(train_path) as f:
        lines = f.readlines()
    x_train = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    y_train = np.zeros((x_train.shape[0], 1))
    train = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)

    with open(test_path) as f:
        lines = f.readlines()
    x_test = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    y_test = np.zeros((x_test.shape[0], 1))
    test = data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=True)


    if val_path is not None:
        with open(train_path) as f:
            lines = f.readlines()
        x_val = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
        x_val = x_val.reshape(x_val.shape[0], 1, 28, 28)
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
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=2)
        self.elu = nn.ELU(alpha=1.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear(z)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.elu(z)
        z = self.conv0(z)
        z = self.elu(z)
        z = self.upsample0(z)
        z = self.conv1(z)
        z = self.elu(z)
        z = self.upsample1(z)
        z = self.conv2(z)
        z = self.elu(z)
        z = self.conv3(z)
        x_tilde = self.sigmoid(z)
        return x_tilde


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mean, logvar = self.encoder(x)
        epsilon = torch.randn_like(logvar)
        z = mean + torch.exp(logvar/2) * epsilon
        x_tilde = self.decoder(z)
        return x_tilde, mean, logvar


class Elbo(nn.Module):
    def __init__(self):
        super(Elbo, self).__init__()

    def forward(self, x, x_tilde, mean, logvar):
        #std = torch.exp(logvar/2)
        #log_encoder = -0.5 * torch.sum(((z - mean)/std)**2,-1) - 0.5 * torch.sum(torch.log(2*np.pi*std**2),-1)
        #log_prior = -0.5 * torch.sum(z ** 2, -1) - 0.5 * torch.sum(torch.log(2 * np.pi), -1)

        x = x.reshape(x.shape[0],-1)
        x_tilde = x_tilde.reshape(x_tilde.shape[0], -1)

        # Analytical KL divergence for a standard normal distribution prior and a
        # multivariate normal distribution with diagonal covariance q(z|x)
        kl = 0.5 * (-logvar - 1 + mean**2 + torch.exp(logvar)).sum(dim=-1)
        log_decoder = (x * torch.log(x_tilde) + (1 - x) * torch.log(1 - x_tilde)).sum(dim=-1) #log likelihood (- cross entropy)
        loss = -(log_decoder - kl).mean()
        return loss


def epoch_train(model, optimizer, loader, loss_fn, epoch, log_interval=None):
    model.train()
    losses = []
    for batch_id, (x, _) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_tilde, mean, logvar = model(x)
        loss = loss_fn(x, x_tilde, mean, logvar)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if log_interval is not None:
            if batch_id%log_interval == 0:
                print (f"----> Epoch {epoch},\t Batch {batch_id},\t Train Loss: {np.mean(np.array(losses)):.2f}")
    return np.mean(np.array(losses))


def epoch_eval(model, loader, loss_fn):
    model.eval()
    losses = []
    for batch_id, (x, _) in enumerate(loader):
        x = x.to(device)
        x_tilde, mean, logvar = model(x)
        loss = loss_fn(x, x_tilde, mean, logvar)
        losses.append(loss.item())
    return np.mean(np.array(losses))


def train(model, optimizer, train_loader, val_loader, loss_fn, epochs, log_interval=None):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        stime = time.time()
        train_loss = epoch_train(model, optimizer, train_loader, loss_fn, epoch, log_interval)
        val_loss = epoch_eval(model, train_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_time = time.time() - stime
        print (f"-> Epoch {epoch},\t Train Loss: {train_loss[-1]:.2f},\t Validation Loss: {val_loss[-1]:.2f},\t "
               f"Epoch Time: {epoch_time} seconds")
    return train_losses, val_losses


seed = 1111
np.random.seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    #Load Data
    train_path = "binarized_mnist_train.amat"
    val_path = "binarized_mnist_val.amat"
    test_path = "binarized_mnist_test.amat"
    train_loader, val_loader, test_loader = load_data(train_path, test_path, val_path, batch_size=4)

    #Create Model
    model = VAE()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = Elbo()
    train(model, optimizer, train_loader, val_loader, loss_fn, epochs=20, log_interval=10)



