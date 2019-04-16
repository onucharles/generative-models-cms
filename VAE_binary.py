import numpy as np
from models import VAE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
import time
import os
import csv
import torchsummary
import io
from contextlib import redirect_stdout


#Load Dataset
def load_data(train_path=None, test_path=None, val_path=None, batch_size=32):
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
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False)

    with open(val_path) as f:
        lines = f.readlines()
    x_val = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
    x_val = x_val.reshape(x_val.shape[0], 1, 28, 28)
    y_val= np.zeros((x_val.shape[0], 1))
    validation = data.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=False)
    return (train_loader, val_loader, test_loader)


#ELBO using Cross Entropy reconstruction loss
class Elbo_CE(nn.Module):
    def __init__(self):
        super(Elbo_CE, self).__init__()

    def forward(self, x, x_tilde_logits, mean, logvar):
        #std = torch.exp(logvar/2)
        #log_encoder = -0.5 * torch.sum(((z - mean)/std)**2,-1) - 0.5 * torch.sum(torch.log(2*np.pi*std**2),-1)
        #log_prior = -0.5 * torch.sum(z ** 2, -1) - 0.5 * torch.sum(torch.log(2 * np.pi), -1)

        x = x.reshape(x.shape[0],-1)
        x_tilde_logits = x_tilde_logits.reshape(x_tilde_logits.shape[0], -1)

        # Analytical KL divergence for a standard normal distribution prior and a
        # multivariate normal distribution with diagonal covariance q(z|x)
        kl = 0.5 * (-logvar - 1 + mean**2 + torch.exp(logvar)).sum(dim=-1)
        log_decoder = -torch.sum(F.binary_cross_entropy_with_logits(input=x_tilde_logits, target=x, reduction="none"), dim=-1)
        #log_decoder = (x * torch.log(x_tilde) + (1 - x) * torch.log(1 - x_tilde)).sum(dim=-1) #log likelihood (- cross entropy)
        loss = -(log_decoder - kl).mean()
        return loss


def epoch_train(model, optimizer, loader, loss_fn, epoch, log_interval=None):
    model.train()
    elbo = []
    for batch_id, (x, _) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_tilde_logits, mean, logvar = model(x)
        loss = loss_fn(x, x_tilde_logits, mean, logvar)
        loss.backward()
        optimizer.step()
        elbo.append(-loss.item())
        if log_interval is not None:
            if batch_id%log_interval == 0:
                print (f"----> Epoch {epoch},\t Batch {batch_id},\t Train ELBO: {np.mean(np.array(elbo)):.2f}")
    return np.mean(np.array(elbo))


def epoch_eval(model, loader, loss_fn):
    model.eval()
    elbo = []
    for batch_id, (x, _) in enumerate(loader):
        x = x.to(device)
        x_tilde_logits, mean, logvar = model(x)
        loss = loss_fn(x, x_tilde_logits, mean, logvar)
        elbo.append(-loss.item())
    return np.mean(np.array(elbo))


def train(model, optimizer, train_loader, val_loader, loss_fn, epochs, save_dir = os.curdir, save_interval=None, log_interval=None):
    train_elbos = []
    val_elbos = []
    epoch_time = []
    max_val_elbo = -1000000
    for epoch in range(epochs):
        stime = time.time()
        train_elbo = epoch_train(model, optimizer, train_loader, loss_fn, epoch, log_interval)
        val_elbo = epoch_eval(model, val_loader, loss_fn)
        train_elbos.append(train_elbo)
        val_elbos.append(val_elbo)
        epoch_time_ = time.time() - stime
        epoch_time.append(epoch_time_)

        if val_elbo > max_val_elbo:
            max_val_elbo = val_elbo
            if save_interval is not None:
                save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, True)
                generate_samples(model, save_dir=save_dir, epoch=epoch, train_samples=train_samples,
                                 val_samples=val_samples, random_z=random_z, best=True)

        if save_interval is not None:
            if epoch % save_interval == 0:
                save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, False)
                generate_samples(model, save_dir=save_dir, epoch=epoch, train_samples=train_samples,
                                 val_samples=val_samples, random_z=random_z)

        print(f"-> Epoch {epoch},\t Train ELBO: {train_elbo:.2f},\t Validation ELBO: {val_elbo:.2f},\t "
              f"Max Validation ELBO: {max_val_elbo:.2f},\t Epoch Time: {epoch_time_:.2f} seconds")
    return train_elbos, val_elbos


def save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, best_model=False):
    if best_model:
        path = os.path.join(save_dir, f'model_best.pt')
    else:
        path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_elbos': train_elbos,
        'val_elbos': val_elbos,
    }, path)

    epochs = [j for j in range(epoch+1)]
    stats = {'Epoch': epochs, 'Train Elbo': train_elbos, "Validation Elbo": val_elbos, "Epoch Time": epoch_time}
    stats_path = os.path.join(save_dir, 'stats.csv')
    with open(stats_path, 'w') as csvfile:
        fieldnames = stats.keys()
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows([[stats[key][j] for key in fieldnames] for j in epochs])


def print_model_summary(model, optimizer, save_dir = None):
    f = io.StringIO()
    with redirect_stdout(f):
        print(optimizer)
        torchsummary.summary(model, (1, 28, 28))
    architechture_summary = f.getvalue()
    print(architechture_summary)

    if save_dir is not None:
        architecture_path = os.path.join(save_dir, 'architecture.txt')
        with open(architecture_path, 'w') as file:
            file.write(architechture_summary)


def generate_samples(model, save_dir, epoch, train_samples, val_samples, random_z, best=False):
    with torch.no_grad():
        train_samples = train_samples.to(device)
        val_samples = val_samples.to(device)
        random_z = random_z.to(device)

        generated_train_samples, mean,_ = model(train_samples)
        generated_train_samples = (F.sigmoid(generated_train_samples)).round().cpu()
        generated_val_samples, _, _ = model(val_samples)
        generated_val_samples = (F.sigmoid(generated_val_samples)).round().cpu()
        generated_random_samples = model.decoder(random_z)
        generated_random_samples = (F.sigmoid(generated_random_samples)).round().cpu()

    if epoch == 0:
        save_image(train_samples, os.path.join(save_dir, "original_train_samples.png"))
        save_image(val_samples, os.path.join(save_dir, "original_val_samples.png"))

    if best:
        generated_train_path = os.path.join(save_dir, f"generated_train_samples_best.png")
        generated_val_path = os.path.join(save_dir, f"generated_val_samples_best.png")
        generated_random_path = os.path.join(save_dir, f"generated_val_samples_best.png")
    else:
        generated_train_path = os.path.join(save_dir, f"generated_train_samples_{epoch}.png")
        generated_val_path = os.path.join(save_dir, f"generated_val_samples_{epoch}.png")
        generated_random_path = os.path.join(save_dir, f"generated_val_samples_{epoch}.png")

    save_image(generated_train_samples, generated_train_path)
    save_image(generated_val_samples, generated_val_path)
    save_image(generated_random_samples, generated_random_path)


batch_size = 64
lr = 3e-4
epochs = 20
save_interval = 3 #save model every save_interval epochs (None for not saving)
log_interval = 100 #print results every log_interval batches (in addition to every epoch (None for not printing)
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
    val_path = "binarized_mnist_valid.amat"
    test_path = "binarized_mnist_test.amat"
    train_loader, val_loader, test_loader = load_data(train_path, test_path, val_path, batch_size=batch_size)

    #Create Save Directory
    save_parent_dir = os.curdir
    model_folder = "VAE_binary_model_"
    previous_folders = [f for f in os.listdir(save_parent_dir) if model_folder in f]
    if not previous_folders:
        save_dir = os.path.join(save_parent_dir, model_folder + f'{1:03d}')
    else:
        last_model = max(previous_folders)
        i = int(last_model.replace(model_folder,"")) + 1
        save_dir = os.path.join(save_parent_dir, model_folder + f'{i:03d}')
    os.mkdir(save_dir)

    #Create Model
    model = VAE()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = Elbo_CE()

    train_samples, _ = next(iter(train_loader))
    val_samples, _ = next(iter(val_loader))
    random_z = torch.randn((batch_size, 100))
    generate_samples(model, save_dir=save_dir, epoch=0, train_samples=train_samples, val_samples=val_samples, random_z=random_z)
    print_model_summary(model, optimizer, save_dir=save_dir)

    train(model, optimizer, train_loader, val_loader, loss_fn, epochs=epochs,
          save_dir=save_dir, save_interval=save_interval, log_interval=log_interval)



