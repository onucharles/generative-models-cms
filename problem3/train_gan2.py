import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
import sys
from models import Critic, Decoder2
from training import Trainer

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
        num_workers=4
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
    critic = Critic(img_size=(32,32,3), dim=16)
    # critic = Critic()
    generator = Decoder2()
    print('critic: ', critic)
    print('generator: ', generator)

    batch_size = 64
    n_epochs = 10
    n_critic_steps = 5  # no of steps to train critic before training generator.
    lr = 1e-4
    z_size = 100
    gp_weight = 10
    print_every = 50

    # get data loaders
    train_loader, valid_loader, test_loader = get_data_loader("data/svhn", batch_size)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-8)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-8)

    trainer = Trainer(generator, critic, generator_optimizer, critic_optimizer, use_cuda=torch.cuda.is_available(),
                      gp_weight=gp_weight, critic_iterations=n_critic_steps, print_every=print_every)
    trainer.train(train_loader, n_epochs, save_training_gif=True)

if __name__ == '__main__':
    main()