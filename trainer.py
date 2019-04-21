import os
from torchvision.datasets import utils
import numpy as np
from models import VAE2, Elbo_Normal
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import time
import os
import csv
import io
from contextlib import redirect_stdout
from math import ceil
from util import load_svhn
import VAE_binary
import argparse
from classify_svhn import get_data_loader
from models import Decoder2, Discriminator
import GAN_svhn

seed = 1111
np.random.seed(seed)
torch.random.manual_seed(seed)
directory = os.getcwd()
print('1 ' + directory)

class Hyperparameters():
    def __init__(self, batch_size=64, lr=1e-4, epochs=21, save_interval=3, log_interval=100, lmbda=-1, discrim_iters=-1):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.save_interval = save_interval  #save model every save_interval epochs (None for not saving)
        self.log_interval = log_interval    #print results every log_interval batches (in addition to every epoch (None for not printing)
        self.lmbda = lmbda                  #Coefficient for gradient discrimination
        self.discrim_iters = discrim_iters  #Number of discriminator iterations per generator iteration when training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generative Models Training')
    parser.add_argument('-m', '--model', default="VAE", type=str,
                        help='Which model to train (VAE or GAN) (default: VAE)')

    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model == "VAE":
        # Hyperparameters for VAE
        hyperparams = Hyperparameters(
            batch_size = 64, 
            lr = 5e-4, 
            epochs = 21, 
            save_interval = 3, 
            log_interval = 100)
        model_folder = (directory + "\\VAE_SVHN_model")

    else:
        # Hyperparameters for GAN
        hyperparams = Hyperparameters(
            batch_size = 64, 
            lr = 1e-4, 
            epochs = 21,
            save_interval = 3, 
            log_interval = 100,
            lmbda = 10,
            discrim_iters = 5)
        model_folder = (directory + "\\GAN_SVHN_model")    

    train, valid, test = get_data_loader("svhn", hyperparams.batch_size)
    train_samples, _ = next(iter(train))
    val_samples, _ = next(iter(valid))
    
    random_z = torch.randn((hyperparams.batch_size, 100))
    random_z_0 = torch.randn((64, 100))
    random_z_1 = torch.randn((64, 100))
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    if args.model == "VAE":
        model = VAE2()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)
        loss_fn = Elbo_Normal()

        train_elbos, val_elbos, epoch_time = VAE_binary.train(model, optimizer, train, valid, loss_fn, epochs=hyperparams.epochs,
                                           save_dir=model_folder, save_interval=hyperparams.save_interval, log_interval=hyperparams.log_interval, 
                                           model_outputs_logits=False, train_samples=train_samples, val_samples=val_samples, random_z=random_z)

        VAE_binary.generate_interpolated_samples(model, model_folder, alphas, interpolate_images=False, random_z_0=random_z_0, random_z_1=random_z_1, 
                                          epoch=hyperparams.epochs, num_samples=64, latent_size=100, model_outputs_logits=False)
        VAE_binary.generate_interpolated_samples(model, model_folder, alphas, interpolate_images=True, random_z_0=random_z_0, random_z_1=random_z_1, 
                                          epoch=hyperparams.epochs, num_samples=64, latent_size=100, model_outputs_logits=False)

    else: #GAN
        generator = Decoder2().to(device)
        discriminator = Discriminator().to(device)

        optimizerGen = optim.Adam(generator.parameters(), lr=hyperparams.lr)
        optimizerDis = optim.Adam(discriminator.parameters(), lr = hyperparams.lr)

        GAN_svhn.GAN_train(generator, discriminator, optimizerGen, optimizerDis, train, hyperparams)
