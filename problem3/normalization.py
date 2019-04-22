"""
This script computes the normalization parameters (mean and standard deviation), per channel,
for the training data (RGB images). So that we can use these values in the data loading pipeline when training our models
"""

import os
import argparse
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import dataset
import random

def main(args):
    """
    Compute normalisation parameters - mean and std.
    """
    traindir = args.data_dir

    trainvalid_dataset = torchvision.datasets.SVHN(
        traindir, split='train',
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

    trainset_size = int(len(trainvalid_dataset) * 0.9)
    train_dataset, validset = dataset.random_split(
        trainvalid_dataset,
        [trainset_size, len(trainvalid_dataset) - trainset_size]
    )
    print('loaded {} training examples of dimension {}'.format(len(train_dataset), train_dataset[0][0].size()))

    # flatten the pixels in each channel. img size is 32 by 32.
    channel0 = np.zeros((len(train_dataset), 32 * 32))
    channel1 = np.zeros((len(train_dataset), 32 * 32))
    channel2 = np.zeros((len(train_dataset), 32 * 32))
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        channel0[i, :] = img[0, :, :].view(-1)
        channel1[i, :] = img[1, :, :].view(-1)
        channel2[i, :] = img[2, :, :].view(-1)


    # take mean and std of each channel
    means = np.mean(channel0), np.mean(channel1), np.mean(channel2)
    stds = np.std(channel0), np.std(channel1), np.std(channel2)
    print('means ', means)
    print('std ', stds)

def test(args):
    """
     Test computed parameters to confirm that data is actually normalised.
    """
    traindir = args.data_dir

    # Mean and std pre-computed from training set.
    # normalize = transforms.Normalize(mean = [0.438, 0.444, 0.473],
    #                                  std = [0.198, 0.201, 0.197])
    normalize = transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))

    trainvalid_dataset = torchvision.datasets.SVHN(
        traindir, split='train',
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    trainset_size = int(len(trainvalid_dataset) * 0.9)
    train_dataset, validset = dataset.random_split(
        trainvalid_dataset,
        [trainset_size, len(trainvalid_dataset) - trainset_size]
    )

    for i in range(len(train_dataset)):
        img = train_dataset[i][0]

        print('mean ', torch.mean(img[0, :, :]))    # mean should be around 0
        print('std ', torch.std(img[0, :, :]))      # std should be around 1

        if i == 100:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'data/svhn')
    args = parser.parse_args()

    seed = 11
    np.random.seed(seed)
    random.seed(seed)

    # main(args)
    test(args)