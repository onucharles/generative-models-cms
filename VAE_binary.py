import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import load_data
from torchvision.utils import save_image
import time
import os
import csv
#import torchsummary
import io
from contextlib import redirect_stdout
from math import ceil
from models import VAE, Elbo_BCE


#K is the number of importance samples
#This functions loads the data from the loader, samples k z_i from the q(z|x_i) for each x_i,
#and calls evaluate_batch_likelihood function to get the likelihood using importance sampling
def estimate_data_likelihood(model, loader, K=200):
    data_log_likelihood = []
    print("Evaluating Likelihood .", end="")
    with torch.no_grad():
        for batch_id, (data, _) in enumerate(loader):
            data = data.to(device)
            mean, logvar = model.encoder(data)
            mean = mean.unsqueeze(1).expand(-1, K, -1)
            logvar = logvar.unsqueeze(1).expand(-1, K, -1)
            z_samples = model.reparametrize(mean, logvar)
            batch_log_likelihood = evaluate_batch_likelihood(model, data.reshape((data.shape[0], -1)), z_samples)
            data_log_likelihood += batch_log_likelihood.tolist()
            print(".", end="")
    print("\n", end="")
    return data_log_likelihood


#Evaluate batch log-likelihood using importance sampling
#x is of size (batch_size M, features_size D)
#z_samples is of size ((batch_size M, importance samples K, latent size L)
def evaluate_batch_likelihood(model, x, z_samples, image_size=(1,28,28)):
    with torch.no_grad():
        z_samples = z_samples.to(device)
        x = x.to(device)

        M = x.shape[0]; D = x.shape[1];
        K = z_samples.shape[1]; L = z_samples.shape[2];

        mean, logvar = model.encoder(x.reshape((M, image_size[0], image_size[1], image_size[2])))
        std = torch.exp(logvar / 2)
        x_tilde_logits = model.decoder(z_samples.reshape((M*K, L)))
        x_tilde_logits = x_tilde_logits.reshape((M, K, D))

        x = x.unsqueeze(1).expand(-1, K, -1)
        mean = mean.unsqueeze(1).expand(-1, K, -1)
        std = std.unsqueeze(1).expand(-1, K, -1)

        log_encoder_zi = -0.5 * torch.sum(((z_samples - mean)/std)**2, -1) - 0.5 * torch.sum(torch.log(2*np.pi*std**2), -1)
        log_prior_zi = -0.5 * torch.sum(z_samples ** 2, -1) - 0.5 * L * np.log(2 * np.pi)
        log_decoder_zi = -torch.sum(F.binary_cross_entropy_with_logits(input=x_tilde_logits, target=x, reduction="none")
                                    , dim=-1)
        log_term = log_decoder_zi + log_prior_zi - log_encoder_zi

        #LogSumExp trick
        max_log_term, _ = log_term.max(dim=-1, keepdim=True)
        batch_log_likelihood = -np.log(K) + max_log_term.squeeze(-1) + (log_term - max_log_term).exp().sum(-1).log()
    return batch_log_likelihood


#Train for one epoch
def epoch_train(model, optimizer, loader, loss_fn, epoch, log_interval=None):
    model.train()
    elbo = []
    for batch_id, (x, _) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()
        x_tilde_logits, mean, logvar = model(x)
        loss = loss_fn(x, x_tilde_logits, mean, logvar)
        loss.backward()
        optimizer.step()
        elbo.append(-loss.item())
        if log_interval is not None:
            if batch_id%log_interval == 0:
                print (f"----> Epoch {epoch},\t Batch {batch_id},\t Train ELBO: {np.mean(np.array(elbo)):.2f}")
    return np.mean(np.array(elbo))


#Evaluate the model on a specific dataloader
def epoch_eval(model, loader, loss_fn):
    model.eval()
    elbo = []
    for batch_id, (x, _) in enumerate(loader):
        x = x.to(device)
        x_tilde_logits, mean, logvar = model(x)
        loss = loss_fn(x, x_tilde_logits, mean, logvar)
        elbo.append(-loss.item())
    return np.mean(np.array(elbo))


#Train the model for a number of epochs
def train(model, optimizer, train_loader, val_loader, loss_fn, epochs, save_dir, save_interval=None, log_interval=None,
          model_outputs_logits=True, train_samples=None, val_samples=None, random_z=None):
    train_elbos = []
    val_elbos = []
    epoch_time = []
    max_val_elbo = -1000000
    for epoch in range(epochs):
        print("Starting epoch %d." % (epoch))
        stime = time.time()
        train_elbo = epoch_train(model, optimizer, train_loader, loss_fn, epoch, log_interval)
        val_elbo = epoch_eval(model, val_loader, loss_fn)
        train_elbos.append(train_elbo)
        val_elbos.append(val_elbo)
        epoch_time_ = time.time() - stime
        epoch_time.append(epoch_time_)

        if val_elbo > max_val_elbo:
            max_val_elbo = val_elbo
            if save_interval is not None:
                save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, True)
                generate_samples(model, save_dir=save_dir, epoch=epoch, train_samples=train_samples,
                                 val_samples=val_samples, random_z=random_z, model_outputs_logits=model_outputs_logits, best=True)
                best_model = model

        if save_interval is not None:
            if epoch % save_interval == 0:
                save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, False)
                generate_samples(model, save_dir=save_dir, epoch=epoch, train_samples=train_samples,
                                 val_samples=val_samples, random_z=random_z, model_outputs_logits=model_outputs_logits)

        print(f"-> Epoch {epoch},\t Train ELBO: {train_elbo:.2f},\t Validation ELBO: {val_elbo:.2f},\t "
              f"Max Validation ELBO: {max_val_elbo:.2f},\t Epoch Time: {epoch_time_:.2f} seconds")
    if save_interval is not None:
        save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, False)
        generate_samples(model, save_dir=save_dir, epoch=epoch, train_samples=train_samples,
                         val_samples=val_samples, random_z=random_z, model_outputs_logits=model_outputs_logits)
    return train_elbos, val_elbos, epoch_time[-1], best_model


#Save the model and the training statistics
def save_model(model, optimizer, train_elbos, val_elbos, epoch_time, epoch, save_dir, best_model=False):
    if best_model:
        path = os.path.join(save_dir, f'model_best.pt')
    else:
        path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_elbos': train_elbos,
        'val_elbos': val_elbos
    }, path)

    epochs = [j for j in range(epoch+1)]
    stats = {'Epoch': epochs, 'Train Elbo': train_elbos, "Validation Elbo": val_elbos, "Epoch Time": epoch_time}
    stats_path = os.path.join(save_dir, 'stats.csv')
    with open(stats_path, 'w') as csvfile:
        fieldnames = stats.keys()
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows([[stats[key][j] for key in fieldnames] for j in epochs])


#Print the model architecture and hyperparameters
def print_model_summary(model, optimizer, save_dir=None, input_size=(1,28,28)):
    f = io.StringIO()
    with redirect_stdout(f):
        print(optimizer)
        torchsummary.summary(model, input_size)
    architecture_summary = f.getvalue()
    print(architecture_summary)

    if save_dir is not None:
        architecture_path = os.path.join(save_dir, 'architecture.txt')
        with open(architecture_path, 'w') as file:
            file.write(architecture_summary)


#Generate original samples vs reconstructed samples for the training and validation sets, plus some random samples
def generate_samples(model, save_dir, epoch, train_samples, val_samples, random_z, model_outputs_logits=True, best=False):
    with torch.no_grad():
        train_samples = train_samples.to(device)
        val_samples = val_samples.to(device)
        random_z = random_z.to(device)

        generated_train_samples, mean,_ = model(train_samples)
        generated_val_samples, _, _ = model(val_samples)
        generated_random_samples = model.decoder(random_z)

        if model_outputs_logits:
            generated_train_samples = (torch.sigmoid(generated_train_samples)).round().cpu()
            generated_val_samples = (torch.sigmoid(generated_val_samples)).round().cpu()
            generated_random_samples = (torch.sigmoid(generated_random_samples)).round().cpu()
        else: #assuming input images are in the range of -1 and 1
            generated_train_samples = ((generated_train_samples + 1) / 2).cpu()
            generated_val_samples = ((generated_val_samples + 1) / 2).cpu()
            generated_random_samples = ((generated_random_samples + 1) / 2).cpu()

    if epoch == 0:
        if model_outputs_logits:
            save_image(train_samples, os.path.join(save_dir, "original_train_samples.png"))
            save_image(val_samples, os.path.join(save_dir, "original_val_samples.png"))
        else:  # assuming input images are in the range of -1 and 1
            save_image((train_samples + 1) / 2, os.path.join(save_dir, "original_train_samples.png"))
            save_image((val_samples + 1) / 2, os.path.join(save_dir, "original_val_samples.png"))

    if best:
        generated_train_path = os.path.join(save_dir, f"generated_train_samples_best.png")
        generated_val_path = os.path.join(save_dir, f"generated_val_samples_best.png")
        generated_random_path = os.path.join(save_dir, f"generated_random_samples_best.png")
    else:
        generated_train_path = os.path.join(save_dir, f"generated_train_samples_{epoch}.png")
        generated_val_path = os.path.join(save_dir, f"generated_val_samples_{epoch}.png")
        generated_random_path = os.path.join(save_dir, f"generated_random_samples_{epoch}.png")

    save_image(generated_train_samples, generated_train_path)
    save_image(generated_val_samples, generated_val_path)
    save_image(generated_random_samples, generated_random_path)


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
            generated_random_samples = model.decoder(random_z)
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
    #if not os.path.isdir(save_dir):
        #os.mkdir(save_dir)
    with torch.no_grad():
        if (random_z_0 is None) or (random_z_1 is None):
            random_z_0 = torch.randn((num_samples, latent_size))
            random_z_1 = torch.randn((num_samples, latent_size))
        random_z_0 = random_z_0.to(device)
        random_z_1 = random_z_1.to(device)
        if interpolate_images:
            generated_images_0 = model.decoder(random_z_0)
            generated_images_1 = model.decoder(random_z_1)
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
                generated_images = model.decoder(random_z)
                if model_outputs_logits:
                    generated_images = (torch.sigmoid(generated_images)).round().cpu()
                else:  # assuming input images are in the range of -1 and 1
                    generated_images = ((generated_images + 1) / 2).cpu()
                generated_random_file_ = generated_random_file + f"_latent_alpha_{alpha}_epoch_{epoch}.png"
                image_path = os.path.join(save_dir, generated_random_file_)
                save_image(generated_images, image_path)




#Hyperparameters
batch_size = 64
lr = 3e-4
epochs = 20
save_interval = 3 #save model every save_interval epochs (None for not saving)
log_interval = 100 #print results every log_interval batches (in addition to every epoch (None for not printing)
seed = 1111
np.random.seed(seed)
torch.random.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    #Load Data
    train_path = "binarized_mnist_train.amat"
    val_path = "binarized_mnist_valid.amat"
    test_path = "binarized_mnist_test.amat"
    train_loader, val_loader, test_loader = load_data(train_path, test_path, val_path, batch_size=batch_size)

    #Create Save Directory
    save_parent_dir = os.curdir
    model_folder = "VAE_binary_model_"
    previous_folders = [f for f in os.listdir(save_parent_dir) if (model_folder in f and os.path.isdir(f))]
    if not previous_folders:
        save_dir = os.path.join(save_parent_dir, model_folder + f'{1:03d}')
    else:
        last_model = max(previous_folders)
        i = int(last_model.replace(model_folder,"")) + 1
        save_dir = os.path.join(save_parent_dir, model_folder + f'{i:03d}')
    os.mkdir(save_dir)

    #Create Model
    model = VAE()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = Elbo_BCE()

    train_samples, _ = next(iter(train_loader))
    val_samples, _ = next(iter(val_loader))
    random_z = torch.randn((batch_size, 100))
    generate_samples(model, save_dir=save_dir, epoch=0, train_samples=train_samples, val_samples=val_samples, random_z=random_z)
    print_model_summary(model, optimizer, save_dir=save_dir)

    #Train Model
    train_elbos, val_elbos, epoch_time = train(model, optimizer, train_loader, val_loader, loss_fn, epochs=epochs,
                                               save_dir=save_dir, save_interval=save_interval, log_interval=log_interval,
                                               train_samples=train_samples, val_samples=val_samples, random_z=random_z)

    #generate num_samples random samples
    # generate_random_samples(model, save_dir, epoch=epochs-1, num_samples=500)

    #Report Results
    test_elbo = epoch_eval(model, test_loader, loss_fn)
    training_log_likelihood = estimate_data_likelihood(model, train_loader)
    training_log_likelihood = sum(training_log_likelihood) / len(training_log_likelihood)
    validation_log_likelihood = estimate_data_likelihood(model, val_loader)
    validation_log_likelihood = sum(validation_log_likelihood) / len(validation_log_likelihood)
    test_log_likelihood = estimate_data_likelihood(model, test_loader)
    test_log_likelihood = sum(test_log_likelihood) / len(test_log_likelihood)
    results_summary = f"Epoch: {epochs - 1}, Train Elbo: {train_elbos[-1]:.2f}, Validation Elbo: {val_elbos[-1]:.2f}, Test Elbo: {test_elbo:.2f}, " \
        f"Training Log Likelihood: {training_log_likelihood:.2f}, Validation Log Likelihood: {validation_log_likelihood:.2f}, " \
        f"Test Log Likelihood: {test_log_likelihood:.2f}, Epoch Time: {epoch_time:.2f}"
    print(results_summary)
    if save_interval is not None:
        results_path = os.path.join(save_dir, 'results.txt')
        with open(results_path, 'w') as file:
            file.write(results_summary)