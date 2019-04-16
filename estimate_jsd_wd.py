import samplers
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models import MLP


torch.set_default_tensor_type(torch.DoubleTensor)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# jensen shannon divergence objective function
class JSD_Loss(nn.Module):
    def __init__(self):
        super(JSD_Loss, self).__init__()

    def forward(self, D_x, D_y):
        E_Dx = torch.mean(torch.log(D_x)) * 0.5
        E_Dy = torch.mean(torch.log(torch.sub(1, D_y))) * 0.5
        constant = torch.Tensor([2]).to(device)
        cost = -1 * (torch.log(constant) + E_Dx + E_Dy)

        return cost

# wasserstein distance objective function
class WD_Loss(nn.Module):
    def __init__(self):
        super(WD_Loss, self).__init__()

    def forward(self, Tx, Ty, grad_Tz, penalty_scale):
        E_Tx = torch.mean(Tx)
        E_Ty = torch.mean(Ty)

        # norm_grad_Tz = torch.norm(grad_Tz, dim=1)
        norm_grad_Tz = torch.sqrt(torch.sum(grad_Tz ** 2, dim=1) + 1e-12)
        E_Tz = torch.mean(torch.pow((norm_grad_Tz - 1), 2))
        cost = -1 * (E_Tx - E_Ty - penalty_scale * E_Tz)

        return cost

# train model to optimise the specified 'criterion'.
def train(model, p_dist, q_dist, optimizer, criterion, device, n_epochs, is_wasserstein):
    losses = []
    for epoch in range(1, n_epochs):
        model.train()
        optimizer.zero_grad()

        # generate a batch of data
        x, y = next(p_dist), next(q_dist)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        # run forward pass
        d_x = model(x)
        d_y = model(y)

        # compute loss and backpropagate.
        if is_wasserstein:
            # compute z
            a = torch.from_numpy(next(iter(samplers.distribution2(x.size(0))))).to(device)
            z = a * x + (1 - a) * y
            z.requires_grad_()      # we need gradients for z.

            # compute Tz
            d_z = model(z)

            # compute grad of d_z wrt z.
            gradients = torch.autograd.grad(outputs=d_z, inputs=z,
                                   grad_outputs=torch.ones(d_z.size(0), 1).to(device),
                                   create_graph=True, retain_graph=True)[0]
            # print('gradients shape: ', gradients.size())

            loss = criterion(d_x, d_y, gradients, penalty_scale=10)
            loss.backward()
        else:
            loss = criterion(d_x, d_y)
            loss.backward()

        # update parameters
        optimizer.step()

        # record the loss for plotting.
        print(loss.item())
        losses.append(loss.item())
    # test(model, p_dist, q_dist, device)

    return losses

# def test(model, p_dist, q_dist, device):
#     model.eval()
#     with torch.no_grad():
#         # generate a batch of data
#         x, y = next(p_dist), next(q_dist)
#         x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
#
#         # run forward pass
#         d_x = model(x)
#         d_y = model(y)
#
#         loss = criterion(d_x, d_y)
#         print('test loss: ', loss.item())

def estimate_divergences(model_hyperparams, lr, criterion, device, batch_size, n_epochs, is_wasserstein):
    input_size, h1_size, h2_size, out_size, out_sigmoid = model_hyperparams
    p_dist = iter(samplers.distribution1(x=0, batch_size=batch_size))

    # train a model for each value of phi
    phi_list = np.linspace(-1, 1, 21)
    jsd_list = []
    for phi in phi_list:
        # create model and optimizer
        model = MLP(input_size, h1_size, h2_size, out_size, out_sigmoid).to(device)
        print(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        q_dist = iter(samplers.distribution1(x=phi, batch_size=batch_size))
        losses = train(model, p_dist, q_dist, optimizer, criterion, device, n_epochs, is_wasserstein)

        # visualise loss.
        # plt.figure()
        # plt.plot(losses)
        # plt.title('phi = {}'.format(phi))
        # plt.show()

        divergence_estimate = -1 * losses[-1]
        print('At phi = {}, divergence estimate = {}'.format(phi, divergence_estimate))
        jsd_list.append(divergence_estimate)

    plt.figure()
    plt.plot(phi_list, jsd_list, 'o')
    plt.xlabel('$\phi$', fontsize=14)
    plt.ylabel('Jensen-Shannon Divergence', fontsize=14)
    plt.show()

def run_jsd():
    input_size = 2
    h1_size = 64
    h2_size = 64
    out_size = 1
    out_sigmoid = True
    batch_size = 512
    n_epochs = 800
    lr = 1e-1
    is_wasserstein = False
    model_hyperparams = input_size, h1_size, h2_size, out_size, out_sigmoid

    # jensen shannon divergence.
    criterion = JSD_Loss()
    estimate_divergences(model_hyperparams, lr, criterion, device, batch_size, n_epochs, is_wasserstein)

def run_wd():
    input_size = 2
    h1_size = 128
    h2_size = 128
    out_size = 1
    out_sigmoid = False
    batch_size = 512
    n_epochs = 600
    lr = 1e-2
    is_wasserstein = True
    model_hyperparams = input_size, h1_size, h2_size, out_size, out_sigmoid

    # Wasserstein divergence.
    criterion = WD_Loss()
    estimate_divergences(model_hyperparams, lr, criterion, device, batch_size, n_epochs, is_wasserstein)

if __name__ == '__main__':
    # run_jsd()
    run_wd()