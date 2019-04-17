import numpy as np
from models import VAE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
import time
import os
import csv
import torchsummary
import io
from contextlib import redirect_stdout
from math import ceil

#Load Dataset
def load_data(train_path=None, test_path=None, val_path=None, batch_size=32, train_val=0.7):
    with open(train_path) as f:
        lines = f.readlines()
    x_train = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
    x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
    y_train = np.zeros((x_train.shape[0], 1))
    train = data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))

    with open(test_path) as f:
        lines = f.readlines()
    x_test = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
    x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))
    y_test = np.zeros((x_test.shape[0], 1))
    test = data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False)

    if val_path is not None:
        with open(val_path) as f:
            lines = f.readlines()
        x_val = np.array([[np.float32(i) for i in line.split(' ')] for line in lines])
        x_val = x_val.reshape((x_val.shape[0], 1, 28, 28))
        y_val= np.zeros((x_val.shape[0], 1))
        validation = data.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
        train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=False)
        return (train_loader, val_loader, test_loader)
    else:

        return (train_loader, val_loader, test_loader)