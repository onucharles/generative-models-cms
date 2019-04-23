from models import Decoder2
import torch
from torchvision.utils import save_image

def get_generator():
    model_path = 'gan_gen_48aa3ed2321d409b81806f21d2c6dec2_epoch_26.pt'
    generator = Decoder2()
    generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    return generator

def generate_samples():
    generator = get_generator()
    for i in range(2): # 2 sets of samples
        latent_samples = generator.sample_latent(64)
        generated_data = generator(latent_samples)
        save_image((generated_data + 1) / 2, f"samples/gan_sample_{i}.png")
    return generated_data

def disentagled_representation():
    generator = get_generator()

    # sample some latent
    latent_samples = generator.sample_latent(5)

    # generate image samples
    generated_data = generator(latent_samples)
    save_image((generated_data + 1) / 2, f"samples/gan_sample_2.png")

    # perturb x random locations of latent and generate sample
    n_loc = 5
    eps = torch.FloatTensor(1).uniform_(-1, 1)              # noise to be added to value at idx.
    for i in range(n_loc):
        idx = torch.randint(0, latent_samples.size(1), (1,))  # location to be perturbed.
        new_latent = latent_samples.clone()
        new_latent[:, idx] = eps
        generated_data = generator(new_latent)
        save_image((generated_data + 1) / 2, f"samples/gan_sample_2(pert{idx.item()}).png")

if __name__ == '__main__':
    # generate_samples()
    disentagled_representation()