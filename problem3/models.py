import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Critic(nn.Module):
    def __init__(self, img_size, dim):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
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
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)

# class Critic(nn.Module):
#     def __init__(self):
#         super(Critic, self).__init__()
#         self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=0)
#         self.batchnorm0 = nn.BatchNorm2d(32)
#         self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0)
#         self.batchnorm1 = nn.BatchNorm2d(64)
#         self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding=0)
#         self.batchnorm2 = nn.BatchNorm2d(256)
#         self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(6, 6), padding=0)
#         self.batchnorm3 = nn.BatchNorm2d(256)
#         # self.linear_mean = nn.Linear(in_features=256, out_features=100, bias=True)
#         # self.linear_logvar = nn.Linear(in_features=256, out_features=100, bias=True)
#         self.elu = nn.ELU(alpha=1.)
#         self.linear = nn.Linear(in_features=256, out_features=1, bias=True)
#
#     def forward(self, x):
#         # print('input to critic: ', x.size())
#         x = self.conv0(x)
#         x = self.elu(x)
#         x = self.batchnorm0(x)
#         x = self.conv1(x)
#         x = self.elu(x)
#         x = self.batchnorm1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.elu(x)
#         x = self.batchnorm2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.elu(x)
#         x = self.batchnorm3(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         # print('output from critic: ', x.size())
#         return x

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
        return x_tilde

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, 100))

# # Model from https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# class Critic(nn.Module):
#     def __init__(self, image_size, image_channel_size, channel_size):
#         # configurations
#         super().__init__()
#         self.image_size = image_size
#         self.image_channel_size = image_channel_size
#         self.channel_size = channel_size
#
#         # layers
#         self.conv1 = nn.Conv2d(
#             image_channel_size, channel_size,
#             kernel_size=4, stride=2, padding=1
#         )
#         self.conv2 = nn.Conv2d(
#             channel_size, channel_size*2,
#             kernel_size=4, stride=2, padding=1
#         )
#         self.conv3 = nn.Conv2d(
#             channel_size*2, channel_size*4,
#             kernel_size=4, stride=2, padding=1
#         )
#         self.conv4 = nn.Conv2d(
#             channel_size*4, channel_size*8,
#             kernel_size=4, stride=1, padding=1,
#         )
#         self.fc = nn.Linear((image_size//8)**2 * channel_size*4, 1)
#
#     def forward(self, x):
#         print('critic input is: {}'.format(x.size()))
#         x = F.leaky_relu(self.conv1(x))
#         x = F.leaky_relu(self.conv2(x))
#         x = F.leaky_relu(self.conv3(x))
#         x = F.leaky_relu(self.conv4(x))
#         x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
#         print('critic output is: {}'.format(x.size()))
#         return self.fc(x)
#
# # Model from https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# class Generator(nn.Module):
#     def __init__(self, z_size, image_size, image_channel_size, channel_size):
#         # configurations
#         super().__init__()
#         self.z_size = z_size
#         self.image_size = image_size
#         self.image_channel_size = image_channel_size
#         self.channel_size = channel_size
#
#         # layers
#         self.fc = nn.Linear(z_size, (image_size//8)**2 * channel_size*8)
#         self.bn0 = nn.BatchNorm2d(channel_size*8)
#         self.bn1 = nn.BatchNorm2d(channel_size*4)
#         self.deconv1 = nn.ConvTranspose2d(
#             channel_size*8, channel_size*4,
#             kernel_size=4, stride=2, padding=1
#         )
#         self.bn2 = nn.BatchNorm2d(channel_size*2)
#         self.deconv2 = nn.ConvTranspose2d(
#             channel_size*4, channel_size*2,
#             kernel_size=4, stride=2, padding=1,
#         )
#         self.bn3 = nn.BatchNorm2d(channel_size)
#         self.deconv3 = nn.ConvTranspose2d(
#             channel_size*2, channel_size,
#             kernel_size=4, stride=2, padding=1
#         )
#         self.deconv4 = nn.ConvTranspose2d(
#             channel_size, image_channel_size,
#             kernel_size=3, stride=1, padding=1
#         )
#
#     def forward(self, z):
#         g = F.relu(self.bn0(self.fc(z).view(
#             z.size(0),
#             self.channel_size*8,
#             self.image_size//8,
#             self.image_size//8,
#         )))
#         g = F.relu(self.bn1(self.deconv1(g)))
#         g = F.relu(self.bn2(self.deconv2(g)))
#         g = F.relu(self.bn3(self.deconv3(g)))
#         g = self.deconv4(g)
#         return F.sigmoid(g)