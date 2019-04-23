import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, img_size, dim):
        super(Critic, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = int(8 * dim * (img_size[0] / 16) * (img_size[1] / 16))
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.linear = nn.Linear(in_features=100, out_features=256, bias=True)
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 5), padding=4)
        self.batchnorm0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=3)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), padding=2)
        self.elu = nn.ELU(alpha=1.)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # print('input to generator: ', z.size())
        z = self.linear(z)
        z = z.unsqueeze(-1).unsqueeze(-1)
        z = self.elu(z)
        z = self.conv0(z)
        z = self.elu(z)
        z = self.batchnorm0(z)
        z = F.interpolate(z, scale_factor=2, mode='bilinear')
        z = self.conv1(z)
        z = self.elu(z)
        z = self.batchnorm1(z)
        z = F.interpolate(z, scale_factor=2, mode='bilinear')
        z = self.conv2(z)
        z = self.elu(z)
        z = self.batchnorm2(z)
        x_tilde = self.conv3(z)
        # print('output from generator: ', x_tilde.size())
        return self.tanh(x_tilde)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, 100))
