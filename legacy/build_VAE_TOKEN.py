import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import torch.nn as nn
import torch.optim as optim
import nnDS as nnDS  # Import the nnDS dataset
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')
from vanillaVAE import VanillaVAE


def train_vae_resp(model, dataloader, dataloader_val, epochs=10, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-12)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    data_length = len(dataloader.dataset)
    for epoch in range(epochs):
        train_loss = 0
        batch_count = 0
        for batch in dataloader:
            batch = batch['past_values'].float().to("cuda")
            optimizer.zero_grad()
            normal_pass = model(batch)
            loss = model.loss_function(*normal_pass)['loss']
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print(f'Epoch {epoch +  1}, Loss: {train_loss / data_length}, batch {batch_count} of {data_length//512}', end='\r')
            batch_count += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        
        for batch in dataloader_val:
            batch = batch['past_values'].float().to("cuda")
            normal_pass = model(batch)
            loss = model.loss_function(*normal_pass)['Reconstruction_Loss']
            val_loss += loss.item()

        model.train()


        #save a state dict
        torch.save(model.state_dict(), 'vae_resp.pt')

        print(f'Epoch {epoch + 1}, Loss: {train_loss / data_length} Val Loss: {val_loss / len(dataloader_val.dataset)}')

if __name__ == '__main__':
    SAMPLE_RATE = 10000 #10kHz sample rate

    # Example usage
    data = nnDS.nnDS("nn_ds_c.joblib", "nn_ds.joblib", 64, 64,  0, tokenizer=nnDS.nnTokenizerPasthrough, dtype=torch.float32)  # Use nnDS dataset
    #split the dataset
    train_set, val_set = torch.utils.data.random_split(data, [len(data)-512, 512])
    # Create the dataloader
    dataloader = DataLoader(train_set, batch_size=512, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(val_set, batch_size=512, shuffle=True, pin_memory=True)

    print(len(data))
    input_dim = 64  # Adjust input_dim based on your dataset
    latent_dim = 3 # Adjust latent_dim based on your dataset
    vae = VanillaVAE(input_dim, latent_dim=latent_dim).to("cuda")
    train_vae_resp(vae, dataloader, dataloader_val, epochs=100, lr=1e-5)

    # Save the model
    torch.save(vae.state_dict(), 'vae_resp.pt')
