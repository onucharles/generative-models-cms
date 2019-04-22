import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from util import load_data
from torchvision.utils import save_image
import time
import os
import csv
import models
import math

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def gradient_penalty(discriminator, x, x_tilde, lmbda):
    epsilon = torch.rand(x.size(0), 1)
    epsilon = epsilon.expand(x.size(0), int(x.nelement()/x.size(0))).contiguous().view(x.size(0), 3, 32, 32)
    epsilon = epsilon.to(device)

    interpolation = epsilon * x + ((1-epsilon)*x_tilde)
    interpolation.to(device)
    interpolation = autograd.Variable(interpolation, requires_grad=True)

    discriminator_interpol = discriminator(interpolation)

    grad = autograd.grad(outputs=discriminator_interpol, inputs=interpolation, 
                        grad_outputs=torch.ones(discriminator_interpol.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad = grad.view(grad.size(0), -1)

    return lmbda*((grad.norm(2, dim=1) - 1) ** 2).mean()

def GAN_train_epoch(generator, discriminator, optimizerGen, optimizerDis, loader, hyperparameters, epoch):
    iteration = 0
    for batch_id, (x, _) in enumerate(loader):
        iteration += 1
        current_batch_size = x.size(0) #To deal with the last batch size being smaller

        discriminator.zero_grad()

        # Loss term from D(real image)
        #x = x.reshape(current_batch_size 3, 32, 32).transpose(0, 2, 3, 1)
        x = x.to(device)

        x_var = autograd.Variable(x)

        disc_x = discriminator(x_var)
        disc_x = disc_x.mean()
        disc_x.backward((torch.FloatTensor([1]) * -1).to(device))

        
        # Loss term from D(fake image)
        noise = torch.randn(current_batch_size, 100)
        noise = noise.to(device)
        noise_var = autograd.Variable(noise)

        x_tilde = autograd.Variable(generator(noise_var).data)

        disc_x_tilde = discriminator(x_tilde)
        disc_x_tilde = disc_x_tilde.mean()
        disc_x_tilde.backward((torch.FloatTensor([1])).to(device))

        # Loss term from Gradient Penalty
        grad_penalty = gradient_penalty(discriminator, x_var, x_tilde, hyperparameters.lmbda)
        grad_penalty.backward()

        disc_loss = disc_x_tilde - disc_x + grad_penalty 

        # Update the Discriminator with the three previous terms
        optimizerDis.step()

        # Update generator every discrim_iters (default every 5 iterations)
        if iteration % hyperparameters.discrim_iters == 0:

            # Lock Discriminator parameters for Generator update
            for params in discriminator.parameters():
                params.requires_grad = False

            generator.zero_grad()

            # Loss term from D(G(z)))
            noise = torch.randn(current_batch_size, 100)
            noise = noise.to(device)
            noise_var = autograd.Variable(noise)

            x_tilde = generator(noise_var)

            gen_cost = discriminator(x_tilde)
            gen_cost = gen_cost.mean()
            gen_cost.backward((torch.FloatTensor([1]) * -1).to(device))
            
            # Loss term
            gen_loss = -gen_cost

            # Update the Generator with the previous loss 
            optimizerGen.step()
            
            # Unlock Discriminator parameters for next iteration
            for params in discriminator.parameters():
                params.requires_grad = True
        
        if hyperparameters.log_interval is not None:
            if (batch_id) % hyperparameters.log_interval == 0 and batch_id != 0:
                print (f"----> Epoch {epoch},\t Batch {batch_id},\t Discriminator Loss: {disc_loss:.2f},\t Generator Loss: {gen_cost:.2f}")

    return disc_loss, gen_loss

def GAN_valid_BCE(generator, discriminator, loader):
    generator.eval()
    discriminator.eval()

    BCE = 0

    for _, (x, _) in enumerate(loader):
        current_batch_size = x.size(0) #To deal with the last batch size being smaller
        x = x.to(device)
        x_real = discriminator(x)

        noise = torch.randn(current_batch_size, 100)
        noise = noise.to(device)
        x_tilde = generator(noise)
        x_fake = discriminator(x_tilde)

        for i in range(current_batch_size):
            if x_real.data[i].item() > 0.5:
                BCE -= math.log(x_real.data[i].item())
            if x_fake.data[i].item() < 0.5:
                BCE -= math.log(1-x_fake.data[i].item())
    return BCE

def GAN_train(generator, discriminator, optimizerGen, optimizerDis, train_loader, valid_loader, hyperparameters, save_dir):
    disc_train_losses = []
    disc_valid_losses = []
    gen_train_losses = []
    epoch_times = []
    min_disc_valid_loss = 1000000
    
    for epoch in range(hyperparameters.epochs):
        print("Starting epoch %d." % (epoch))
        stime = time.time()
        disc_train_loss, gen_train_loss = GAN_train_epoch(generator, discriminator, optimizerGen, optimizerDis, train_loader, hyperparameters, epoch)
        disc_valid_loss = GAN_valid_BCE(generator, discriminator, valid_loader)

        disc_train_losses.append(disc_train_loss)
        gen_train_losses.append(gen_train_loss)
        disc_valid_losses.append(disc_valid_loss)
        epoch_time = time.time() - stime
        epoch_times.append(epoch_time)

        if disc_valid_loss < min_disc_valid_loss:
            min_disc_valid_loss = disc_valid_loss
            if hyperparameters.save_interval is not None:
                save_GAN(generator, discriminator, optimizerGen, optimizerDis, disc_train_losses, 
                disc_valid_losses, gen_train_losses, epoch_times, epoch, save_dir, True)

        if hyperparameters.save_interval is not None:
            if epoch % hyperparameters.save_interval == 0:
                save_GAN(generator, discriminator, optimizerGen, optimizerDis, disc_train_losses, 
                disc_valid_losses, gen_train_losses, epoch_times, epoch, save_dir, False)
        
        print(f"-> Epoch {epoch},\t WGAN-GP Train Loss: {disc_train_loss:.2f},\t BCE Validation Loss: {disc_valid_loss:.2f},\t "
              f"Min Validation Loss: {min_disc_valid_loss:.2f},\t Epoch Time: {epoch_time:.2f} seconds")
    
    if hyperparameters.save_interval is not None:
        save_GAN(generator, discriminator, optimizerGen, optimizerDis, disc_train_losses, 
        disc_valid_losses, gen_train_losses, epoch_times, epoch, save_dir, False)

#Save the model and the training statistics
def save_GAN(generator, discriminator, optimizerGen, optimizerDis, disc_train_losses, 
             disc_valid_losses, gen_train_losses, epoch_times, epoch, save_dir, best_model=False):
    if best_model:
        path = os.path.join(save_dir, f'model_best.pt')
    else:
        path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
    
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizerGen_state_dict': optimizerGen.state_dict(),
        'optimizerDis_state_dict': optimizerDis.state_dict(),
        'disc_train_losses': disc_train_losses,
        'disc_valid_losses': disc_valid_losses,
        'gen_train_losses': gen_train_losses
    }, path)

    epochs = [j for j in range(epoch+1)]
    stats = {'Epoch': epochs, 
             'Discriminator Train Loss': disc_train_losses, 
             'Discriminator Validation Loss': disc_valid_losses, 
             'Generator Train Loss': gen_train_losses, 
             'Epoch Time': epoch_times}
    stats_path = os.path.join(save_dir, 'stats.csv')
    with open(stats_path, 'w') as csvfile:
        fieldnames = stats.keys()
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows([[stats[key][j] for key in fieldnames] for j in epochs])
