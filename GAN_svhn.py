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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def gradient_penalty(discriminator, x, x_tilde, batch_size):
    epsilon = torch.rand(batch_size, 1)
    epsilon = epsilon.expand(batch_size, x.nelement()/batch_size).contiguous().view(batch_size, 3, 32, 32)
    epsilon = epsilon.to(device)

    interpolation = epsilon * x + ((1-epsilon)*x_tilde)
    interpolation.to(device)

    discriminator_interpol = discriminator(interpolation)

    grad = autograd.grad(outputs=discriminator_interpol, inputs=interpolation, 
                        grad_outputs=torch.ones(discriminator_interpol.size().to(device)),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    grad = grad.view(grad.size(0), -1)

    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

def GAN_train(generator, discriminator, optimizerGen, optimizerDis, loader, hyperparameters):
    iteration = 0
    for epoch in range(hyperparameters.epochs):
        for batch_id, (x, _) in enumerate(loader):
            iteration += 1

            discriminator.zero_grad()

            # Train with Real image
            #x = x.reshape(hyperparameters.batch_size, 3, 32, 32).transpose(0, 2, 3, 1)
            x = x.to(device)
            print(x.size())

            x_var = autograd.Variable(x)

            disc_x = discriminator(x_var)
            disc_x = disc_x.mean()
            disc_x.backward((torch.FloatTensor([1]) * -1).to(device))

            
            # Train with Generated image
            noise = torch.randn(hyperparameters.batch_size, 100)
            noise = noise.to(device)
            noise_var = autograd.Variable(noise, volatile=True)

            x_tilde = autograd.Variable(generator(noise_var).data)

            disc_x_tilde = discriminator(x_tilde)
            disc_x_tilde = disc_x_tilde.mean()
            disc_x_tilde.backward((torch.FloatTensor([1])).to(device))

            # Train with Gradient Penalty
            grad_penalty = gradient_penalty(discriminator, x_var.data, x_tilde.data, hyperparameters.batch_size)
            grad_penalty.backward()

            disc_cost = disc_x_tilde - disc_x + hyperparameters.lmbda * grad_penalty 

            optimizerDis.step()

            # Update generator every discrim_iters
            if iteration % hyperparameters.discrim_iters == 0:
                for params in discriminator.parameters():
                    params.requires_grad = False

                generator.zero_grad()

                noise = torch.randn(hyperparameters.batch_size, 100)
                noise = noise.to(device)
                noise_var = autograd.Variable(noise)

                x_tilde = generator(noise_var)

                gen_cost = discriminator(x_tilde)
                gen_cost = gen_cost.mean()
                gen_cost.backward((torch.FloatTensor([1]) * -1).to(device))
                gen_cost = -gen_cost

                optimizerGen.step()
                
                for params in discriminator.parameters():
                    params.requires_grad = True

            if hyperparameters.log_interval is not None:
                if batch_id%hyperparameters.log_interval == 0:
                    print (f"----> Epoch {epoch},\t Batch {batch_id},\t Disc_cost: {disc_cost}")
