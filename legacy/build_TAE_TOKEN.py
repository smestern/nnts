import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import torch.nn as nn
import torch.optim as optim
from nnDS import nnDS, nnTokenizerPasthrough  # Import the nnDS dataset
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class TAE(nn.Module):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = 2048,
                 **kwargs) -> None:
        super(TAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_dim = in_channels

        encoder_layers = nn.TransformerEncoderLayer(d_model=in_channels, dim_feedforward=hidden_dims, nhead=8, batch_first=True)
        

        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.fc_mu = nn.Linear(in_channels, latent_dim)
        self.fc_var = nn.Linear(in_channels, latent_dim)


        # Build Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dims, dim_feedforward=hidden_dims, nhead=8, batch_first=True)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims)

       
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=6)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims, self.in_dim),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input.unsqueeze(1))
        #result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var, result]

    def decode(self, z: Tensor, memory: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 64)
        result = self.decoder(result, memory)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var, memory = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z,  memory), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight =  0.000025 #kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = torch.nn.functional.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def train_vae(model, dataloader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-12)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            batch = batch['past_values'].float().to("cuda")
            optimizer.zero_grad()
            normal_pass = model(batch)
            loss = model.loss_function(*normal_pass)['loss']
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')

if __name__ == '__main__':
    # Example usage
    data = nnDS("nn_ds_c.joblib", "nn_ds.joblib", 1024, 1024,  0, length=512*1000, tokenizer=nnTokenizerPasthrough, dtype=torch.float32)  # Use nnDS dataset
    dataloader = DataLoader(data, batch_size=512)
    print(len(data))
    input_dim = 64  # Adjust input_dim based on your dataset
    latent_dim = 8 # Adjust latent_dim based on your dataset
    vae = TAE(input_dim, latent_dim=latent_dim).to("cuda")
    train_vae(vae, dataloader, epochs=50)