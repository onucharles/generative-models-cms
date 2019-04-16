#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""


from __future__ import print_function
import numpy as np
import torch 
import matplotlib.pyplot as plt
import samplers
import torch.nn as nn
from models import MLP, MLP2

torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)

# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4'' 
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

# our loss function
class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()

    def forward(self, D_x, D_y):
        E_Dx = torch.mean(torch.log(D_x))
        E_Dy = torch.mean(torch.log(1 - D_y))
        cost = -1 * (E_Dx + E_Dy)
        return cost

# train model to optimise specified 'criterion'.
def train(model, f1, f0, optimizer, criterion, device, n_epochs):
    losses = []
    for epoch in range(1, n_epochs):
        model.train()
        optimizer.zero_grad()

        # generate a batch of data
        x, y = next(f1), next(f0)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        # run forward pass
        d_x = model(x)
        d_y = model(y)

        # compute loss and backpropagate.
        loss = criterion(d_x, d_y)
        loss.backward()

        # update parameters
        optimizer.step()

        # record the loss
        print(loss.item())
        losses.append(loss.item())

    return losses

# train discriminator between standard gaussian and distribution4.
def train_discriminator():
    input_size = 1
    n_hidden = 3
    n_units = 256
    out_size = 1
    out_sigmoid = True
    batch_size = 512
    n_epochs = 800
    lr = 1e-2

    # create model, training criterion and optimizer.
    criterion = ValueFunction()
    model = MLP2(input_size, n_hidden, n_units, out_size, out_sigmoid).to(device)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # get iterators for distributions
    f1 = iter(samplers.distribution4(batch_size))
    f0 = iter(samplers.distribution3(batch_size))
    losses = train(model, f1, f0, optimizer, criterion, device, n_epochs)

    # visualise loss.
    plt.figure()
    plt.plot(losses)
    # plt.show()
    return model

model = train_discriminator()


############### plotting things
############### (1) plot the output of your trained discriminator 
############### (2) plot the estimated density contrasted with the true density

r = model(torch.from_numpy(xx.reshape(len(xx),1)).to(device)) # evaluate xx using your discriminator; replace xx with the output
r = r.detach().cpu().numpy().flatten()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
# plt.title(r'$D(x)$', fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$D(x)$', fontsize=14)

# estimate the density of distribution4 (on xx) using the discriminator;
# replace "np.ones_like(xx)*0." with your estimate
estimate = N(xx) * r / (1 - r)

plt.subplot(1,2,2)
plt.plot(xx,estimate, label='Estimated')
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx), label='True')
plt.legend()
# plt.title('Estimated and True Densities for $f_1$', fontsize=14)
plt.xlabel('$x$', fontsize=14)
plt.ylabel('density $f_1(x)$', fontsize=14)
plt.show()










