from comet_ml import Experiment
import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# adapted from https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
class Trainer():
    def __init__(self, args, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': [], 'crossent': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        
        # create samples directory
        self.experiment = Experiment(api_key="w7QuiECYXbNiOozveTpjc9uPg", project_name="project3",
                                workspace="ift6135final")
        #exp_id = current_datetime()
        exp_id = self.experiment.id
        self.samples_dir = f"experiment/{exp_id}/"
        create_folder(self.samples_dir)

        # create generator and critic model directories
        create_folder(f"experiment/{exp_id}/generator")
        create_folder(f"experiment/{exp_id}/critic")

        # save config params to file
        save_json(vars(args), f"{self.samples_dir}/config.json")
        self.args = args
        self.experiment.log_parameters(vars(args))

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size(0)
        generated_data = self.sample_generator(batch_size)

        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, (x, _) in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(x)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(x)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))
                    self.experiment.log_metric('generator_loss', self.losses['G'][-1])
                # print("Cross entropy loss: {}".format(self.losses['crossent'][-1]))
                self.experiment.log_metric('critic_loss', self.losses['D'][-1])
                self.experiment.log_metric('gradient_norm', self.losses['gradient_norm'][-1])


    def train(self, data_loader, epochs, save_training_gif=True, save_model=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(64))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            # training_progress_images = []

        # save 1 batch of original images.
        for i, (x, _) in enumerate(data_loader):
            if i == 0:
                save_image((x.cpu() + 1) / 2, f"{self.samples_dir}/original_sample_{i}.png")
                break

        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            if save_model:
                torch.save(self.G.state_dict(), f"{self.samples_dir}/generator/epoch_{epoch}.pt")
                torch.save(self.D.state_dict(), f"{self.samples_dir}/critic/epoch_{epoch}.pt")
            if save_training_gif:
                save_image((self.G(fixed_latents).cpu() + 1) / 2, f"{self.samples_dir}/generated_train_{epoch}.png")

    def sample_generator(self, num_samples):
        # latent_samples = Variable(self.G.sample_latent(num_samples))
        latent_samples = Variable(self.G.sample_latent(num_samples))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples)
        return generated_data

import os
import time
import json
def create_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("created directory: " + str(newpath))

def save_json(data, file_path):
    with open(file_path, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data
def current_datetime():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr
