import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

class Generator(nn.Module):
    def __init__(self, latent_dim, iter, chunk_size, num_chunks):
        super(Generator, self).__init__()

        self.mu = nn.Parameter(data=torch.randn(num_chunks, latent_dim), requires_grad=True)
        self.sigma = nn.Parameter(data=torch.randn(num_chunks, latent_dim), requires_grad=True)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 200, normalize=True),
            *block(200, 200),
            nn.Linear(200, chunk_size),
        )

    def forward(self, z, chunks):
        # Concatenate label embedding and image to produce input
        # g_input = torch.cat((self.chunk_emb_learnable[chunks], z), -1)
        g_input = self.mu[chunks] + torch.log(1 + torch.exp(self.sigma[chunks])) * z
        fake_param = self.model(g_input)
        return fake_param


class Discriminator(nn.Module):
    def __init__(self, iter, chunk_size, num_chunks):
        super(Discriminator, self).__init__()

        self.chunk_emb_learnable = nn.Parameter(data=torch.randn(num_chunks, 256), requires_grad=True)

        # Copied from cgan.py
        self.model = nn.Sequential(
            nn.Linear(chunk_size + 256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, chunks):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.chunk_emb_learnable[chunks]), -1)
        validity = self.model(d_in)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples, chunks, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).to(device)
    chunks = chunks.to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, chunks)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

