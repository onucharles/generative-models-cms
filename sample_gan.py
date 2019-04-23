import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision.utils import save_image
from math import ceil

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

#Generate a num_samples random samples
def generate_random_samples(model, save_dir, random_z_samples=None, epoch=-1, num_samples=200, latent_size=100,
                            model_outputs_logits=True, samples_per_image=64, generated_random_file="generated_random_samples"):
    with torch.no_grad():
        num_images = ceil(num_samples / samples_per_image)
        for k in range(1, num_images +1):
            if k == num_images and num_samples%samples_per_image != 0:
                samples_per_image = num_samples%samples_per_image
            if random_z_samples is None:
                random_z = torch.randn((samples_per_image, latent_size))
            else:
                try:
                    random_z = random_z_samples[(k-1)*samples_per_image:k*samples_per_image]
                except:
                    if len(random_z_samples) < k*samples_per_image:
                        print("random_z_samples are too small for the number of images requested")
            random_z = random_z.to(device)
            generated_random_samples = model(random_z)
            if model_outputs_logits:
                generated_random_samples = (torch.sigmoid(generated_random_samples)).round().cpu()
            else: #assuming input images are in the range of -1 and 1
                generated_random_samples = ((generated_random_samples + 1) / 2).cpu()
            generated_random_file_ = generated_random_file + f"_epoch_{epoch}_numsamples_{num_samples}_{k:03d}.png"
            image_path = os.path.join(save_dir, generated_random_file_)
            save_image(generated_random_samples, image_path)


#Generate an interpolated image between two images generated given two latent variables
#random_z_1 and random_z_2 with factor alpha x = alpha*x_0 + (a-alpha)*x_1
def generate_interpolated_samples(model, save_dir, alphas, interpolate_images=False, random_z_0=None, random_z_1=None, epoch=-1, num_samples=64, latent_size=100,
                            model_outputs_logits=False, generated_random_file="generated_random_samples_interpolated"):
    save_dir = os.path.join(save_dir, "interpolation_samples")
    with torch.no_grad():
        if (random_z_0 is None) or (random_z_1 is None):
            random_z_0 = torch.randn((num_samples, latent_size))
            random_z_1 = torch.randn((num_samples, latent_size))
        random_z_0 = random_z_0.to(device)
        random_z_1 = random_z_1.to(device)
        if interpolate_images:
            generated_images_0 = model(random_z_0)
            generated_images_1 = model(random_z_1)
            for alpha in alphas:
                generated_images = alpha*generated_images_0 + (1.-alpha) * generated_images_1
                if model_outputs_logits:
                    generated_images = (torch.sigmoid(generated_images)).round().cpu()
                else:  # assuming input images are in the range of -1 and 1
                    generated_images = ((generated_images + 1) / 2).cpu()
                generated_random_file_ = generated_random_file + f"_images_alpha_{alpha}_epoch_{epoch}.png"
                image_path = os.path.join(save_dir, generated_random_file_)
                save_image(generated_images, image_path)
        else:
            for alpha in alphas:
                random_z = alpha*random_z_0 + (1.-alpha)*random_z_1
                generated_images = model(random_z)
            
                if model_outputs_logits:
                    generated_images = (torch.sigmoid(generated_images)).round().cpu()
                else:  # assuming input images are in the range of -1 and 1
                    generated_images = ((generated_images + 1) / 2).cpu()
                generated_random_file_ = generated_random_file + f"_latent_alpha_{alpha}_epoch_{epoch}.png"
                image_path = os.path.join(save_dir, generated_random_file_)
                save_image(generated_images, image_path)

seed = 1111
torch.random.manual_seed(seed)
directory = os.getcwd()
save_dir = (directory + "\\GAN_SVHN_model")
sample_folder = (save_dir + "\\samples\\samples\\")    

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

gan = Decoder2()
gan.to(device)

gan.load_state_dict(torch.load('best_gan.pt'))
gan.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generative Models Training')
    parser.add_argument('-m', '--mode', default="interp", type=str,
                        help='Which mode to run (sample or interp) (default: samples)')
    args = parser.parse_args()

    if args.mode == "samples":
        generate_random_samples(gan, sample_folder, num_samples=1000, model_outputs_logits=False, samples_per_image=1, generated_random_file='gan')

    if args.mode == "interp": #args.mode == "interp":
        random_z = torch.randn((64, 100))
        random_z_0 = torch.randn((64, 100))
        random_z_1 = torch.randn((64, 100))
        alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        generate_interpolated_samples(gan, save_dir, alphas, interpolate_images=False, random_z_0=random_z_0, random_z_1=random_z_1, 
                                           num_samples=64, latent_size=100, model_outputs_logits=False)

        generate_interpolated_samples(gan, save_dir, alphas, interpolate_images=True, random_z_0=random_z_0, random_z_1=random_z_1, 
                                           num_samples=64, latent_size=100, model_outputs_logits=False)