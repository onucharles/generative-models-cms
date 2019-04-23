import torch
import torch.nn as nn
from models import Critic, Decoder2

class WGAN_GP(nn.Module):
    def __init__(self, z_size, penalty_strength, img_size, img_channel_size, c_channel_size, g_channel_size, device):
        super().__init__()
        self.z_size = z_size
        self.penalty_strength = penalty_strength
        self.img_size = img_size
        self.img_channel_size = img_channel_size
        self.critic = Critic(img_size=(32,32,3), dim=16)
        self.generator = Decoder2()
        self.device = device

    def critic_loss(self, real_data):
        batch_size = real_data.size(0)
        fake_data = self.generator(self.sample_noise(batch_size))
        d_x = torch.mean(self.critic(real_data))
        d_y = torch.mean(self.critic(fake_data))
        grad_penalty = self._gradient_penalty(real_data, fake_data)
        loss = -(d_x - d_y - grad_penalty)
        return loss

    def generator_loss(self, z):
        loss = -torch.mean(self.critic(self.generator(z)))
        return loss

    def sample_noise(self, batch_size):
        return torch.randn((batch_size, self.z_size)).to(self.device)

    def sample_image(self, batch_size):
        return self.generator(self.sample_noise(batch_size))

    def _gradient_penalty(self, x, y):
        assert x.size() == y.size()
        batch_size = x.size(0)

        a = torch.rand(x.size()).to(self.device)
        # print('z is: ', a.size())
        # print('x is: ', x.size())
        # print('y is: ', y.size())
        z = a * x + (1 - a) * y
        z.requires_grad_()  # we need gradients for z.

        # apply critic on interpolation of x and y
        d_z = self.critic(z)

        # compute grad of d_z wrt z.
        gradients = torch.autograd.grad(outputs=d_z, inputs=z,
                                        grad_outputs=torch.ones(d_z.size(0), 1).to(self.device),
                                        create_graph=True, retain_graph=True)[0]

        norm_grad = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # print('gradient norm is: ', norm_grad.mean().item())
        grad_penalty = self.penalty_strength * torch.mean(torch.pow((norm_grad - 1), 2))

        return grad_penalty
