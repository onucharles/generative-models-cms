import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
from wgan_gp import WGAN_GP
import sys

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])

def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    print('training set size: ', trainset_size)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader

def main():
    # initialise parameters
    batch_size = 16
    n_epochs = 10
    n_critic_steps = 2      # no of steps to train critic before training generator.
    lr = 0.001
    z_size = 100
    img_size = 32
    img_channel_size = 3
    c_channel_size = 64
    g_channel_size = 64
    penalty_strength = 10

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # create GAN
    wgan = WGAN_GP(z_size, penalty_strength, img_size, img_channel_size, c_channel_size, g_channel_size, device).to(device)
    print('Critic summary: ', wgan.critic)
    print('Generator summary: ', wgan.generator)

    # create optimizers
    critic_optimizer = torch.optim.Adam(wgan.critic.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    generator_optimizer = torch.optim.Adam(wgan.generator.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)

    # get data loaders
    train_loader, valid_loader, test_loader = get_data_loader("data/svhn", batch_size)

    # train
    step_no = 0
    critic_losses = []
    generator_losses = []
    for epoch in range(n_epochs):
        wgan.train()
        for i, (x, _) in enumerate(train_loader):
            step_no += 1
            print('Step: {}----'.format(step_no))
            x = x.to(device)

            # train critic
            critic_optimizer.zero_grad()
            critic_loss = wgan.critic_loss(x)
            critic_loss.backward()
            critic_optimizer.step()

            # save critic loss
            print('Critic loss: {}'.format(critic_loss.item()))
            critic_losses.append(critic_loss.item())

            # train generator every 'n_critic_steps' times.
            if step_no % n_critic_steps == 0:
                generator_optimizer.zero_grad()
                z = wgan.sample_noise(batch_size)
                generator_loss = wgan.generator_loss(z)
                generator_loss.backward()
                generator_optimizer.step()

                # save generator loss
                print('Generator loss: {}'.format(generator_loss.item()))
                generator_losses.append(generator_loss.item())
                sys.exit(0)
                # generate and save images.

        # save checkpoint after every x epochs.


if __name__=='__main__':
    main()