from comet_ml import Experiment
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
import sys
from models import Critic, Decoder2
from training import Trainer
import argparse

image_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.438, 0.444, 0.473],
    #                      std = [0.198, 0.201, 0.197])
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
        num_workers=12
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--n_critic_steps', type=int, default=5)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--c_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--c_load_model_path', type=str, default=None)
    parser.add_argument('--g_load_model_path', type=str, default=None)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    args = parser.parse_args()

    # create critic and generator
    critic = Critic(img_size=(32,32,3), dim=16)
    # critic = Critic()
    generator = Decoder2()
    print('critic: ', critic)
    print('generator: ', generator)

    # load saved model if provided
    if not args.c_load_model_path is None and not args.g_load_model_path is None:
        critic.load_state_dict(torch.load(args.c_load_model_path, map_location=lambda storage, loc: storage))
        generator.load_state_dict(torch.load(args.g_load_model_path, map_location=lambda storage, loc: storage))

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    n_critic_steps = args.n_critic_steps  # no of steps to train critic before training generator.
    g_lr = args.g_lr
    c_lr = args.c_lr
    z_size = 100
    gp_weight = 10
    print_every = args.print_every
    save_model = args.save_model
    beta1 = args.beta1
    beta2 = args.beta2

    # get data loaders
    train_loader, valid_loader, test_loader = get_data_loader("data/svhn", batch_size)

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=c_lr, betas=(beta1, beta2), weight_decay=0)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2), weight_decay=0)

    trainer = Trainer(args, generator, critic, generator_optimizer, critic_optimizer, use_cuda=torch.cuda.is_available(),
                      gp_weight=gp_weight, critic_iterations=n_critic_steps, print_every=print_every)
    trainer.train(train_loader, n_epochs, save_training_gif=True, save_model=save_model)

if __name__ == '__main__':
    main()
