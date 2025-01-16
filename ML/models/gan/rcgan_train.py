# This Python file is a training script for the RCGAN (Regressional and Conditional GAN) model

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import utils
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
from train_config import *
import time


noisy_dim = config["rcgan_cellular"]["noise_size"]
lr_g = config["rcgan_cellular"]["lr_g"]
lr_d = config["rcgan_cellular"]["lr_d"]
lr_r = config["rcgan_cellular"]["lr_r"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
d_dim = config["rcgan_cellular"]["d_dim"]


# Optimizers
# Initialize the generator, discriminator, and regressor models and move them to the device (GPU if available)
generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
discriminator = Discriminator(cellular_gan_input_dim, cellular_gan_output_dim).to(device)
regressor = Regressor(cellular_gan_input_dim, cellular_gan_output_dim).to(device)


optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
optimizer_r = optim.Adam(regressor.parameters(), lr=lr_r)


# Learning rate schedulers for adjusting the learning rate during training
scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min', factor=0.2, patience=3, verbose=True, min_lr=1e-20)
scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, 'min', factor=0.2, patience=3, verbose=True, min_lr=1e-20)
scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, 'min', factor=0.2, patience=3, verbose=True, min_lr=1e-20)


# Initialize variables to keep track of the minimum loss
best_g_loss = float('inf')
best_d_loss = float('inf')
best_r_loss = float('inf')


# Lists to store all loss values
D_Losses, G_Losses, R_Losses = [], [], []
D_real_accuracies, D_fake_accuracies = [], []


start_time = time.time()


for epoch in range(epochs):
    # Initialize losses for each epoch
    D_Losses_ = []
    G_Losses_ = []
    R_Losses_ = []
    D_real_accuracies_ = []
    D_fake_accuracies_ = []


    # Set models to training mode
    generator.train()
    discriminator.train()
    regressor.train()


    # Training loop using tqdm for progress tracking
    with tqdm(enumerate(cellular_gan_train), total=len(cellular_gan_train), desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for batch_idx, (inputs, cond) in pbar:
            # Organize data
            inputs = inputs.to(device)
            cond = cond.to(device)
            cond = cond.to(torch.float32)
            inputs = inputs.to(torch.float32)


            batch_size = inputs.size(0)
            batch_y = cond
            batch_z = torch.randn(batch_size, noisy_dim).to(device)
            d = batch_y[:, 0:1]

            # ---- Train Generator ----
            # Zero gradients for the generator optimizer
            optimizer_g.zero_grad()

            # Generate fake data
            fake_data = generator(batch_z, batch_y)
            latent_fake, fake_pred = regressor(fake_data)
            latent_real, real_pred = regressor(inputs)

            # Calculate the generator loss
            G_loss = F.mse_loss(fake_pred, batch_y) + F.mse_loss(fake_data, inputs)
            G_loss.backward(retain_graph=True)
            optimizer_g.step()

            G_Losses_.append(G_loss.item())

            # ---- Train Discriminator ----
            # Zero gradients for the discriminator optimizer
            optimizer_d.zero_grad()


            real_data_pred = discriminator(latent_real, batch_y)
            fake_data_pred = discriminator(latent_fake.detach(), batch_y)


            # Calculate discriminator loss for real and fake data
            D_real_loss = F.mse_loss(real_data_pred, torch.ones_like(real_data_pred) * 0.9)
            D_fake_loss = F.mse_loss(fake_data_pred, torch.zeros_like(fake_data_pred))
            D_loss = (D_real_loss + D_fake_loss) / 2
            D_loss.backward(retain_graph=True)
            optimizer_d.step()

            D_Losses_.append(D_loss.item())

            # Calculate discriminator accuracy
            real_accuracy = ((real_data_pred > 0.5).float() == torch.ones_like(real_data_pred)).float().mean().item()
            fake_accuracy = ((fake_data_pred < 0.5).float() == torch.zeros_like(fake_data_pred)).float().mean().item()

            D_real_accuracies_.append(real_accuracy)
            D_fake_accuracies_.append(fake_accuracy)

            # ---- Train Regressor ----
            # Zero gradients for the regressor optimizer
            optimizer_r.zero_grad()

            # Calculate the regressor loss
            R_loss = F.mse_loss(real_pred, batch_y)
            R_loss.backward()
            optimizer_r.step()

            R_Losses_.append(R_loss.item())

            # Update the progress bar
            pbar.set_postfix(
                D_Loss=np.mean(D_Losses_),
                G_Loss=np.mean(G_Losses_),
                R_Loss=np.mean(R_Losses_)
            )


    # Calculate average losses and accuracies for the epoch
    avg_D_loss = np.mean(D_Losses_)
    avg_G_loss = np.mean(G_Losses_)
    avg_R_loss = np.mean(R_Losses_)

    avg_real_accuracy = np.mean(D_real_accuracies_)
    avg_fake_accuracy = np.mean(D_fake_accuracies_)

    D_real_accuracies.append(avg_real_accuracy)
    D_fake_accuracies.append(avg_fake_accuracy)

    D_Losses.append(avg_D_loss)
    G_Losses.append(avg_G_loss)
    R_Losses.append(avg_R_loss)

    # Save the generator model if the current loss is lower than the best loss
    if avg_G_loss < best_g_loss:
        best_g_loss = avg_G_loss
        # Save the generator
        torch.save(generator.state_dict(), best_g_model_path_cellular)
        print(f"Best Generator model saved with loss: {avg_G_loss:.4f}")

    # Save the discriminator model if the current loss is lower than the best loss
    if avg_D_loss < best_d_loss:
        best_d_loss = avg_D_loss
        # Save the discriminator
        torch.save(discriminator.state_dict(), best_d_model_path_cellular)
        print(f"Best Discriminator model saved with loss: {avg_D_loss:.4f}")

    # Save the regressor model if the current loss is lower than the best loss
    if avg_R_loss < best_r_loss:
        best_r_loss = avg_R_loss
        # Save the regressor
        torch.save(regressor.state_dict(), best_r_model_path_cellular)
        print(f"Best Regressor model saved with loss: {avg_R_loss:.4f}")

    # Print loss information
    print(f"Epoch {epoch+1}/{epochs} | "
          f"G Loss: {avg_G_loss:.4f} | "
          f"D Loss: {avg_D_loss:.4f} | "
          f"R Loss: {avg_R_loss:.4f} | "
          f"D Real Accuracy: {avg_real_accuracy:.4f} | "
          f"D Fake Accuracy: {avg_fake_accuracy:.4f}")

    # Update the learning rate schedulers
    scheduler_g.step(avg_G_loss)  # Update the learning rate using the generator loss
    scheduler_d.step(avg_D_loss)  # Update the learning rate using the discriminator loss
    scheduler_r.step(avg_R_loss)  # Update the learning rate using the regressor loss

# Training complete
print('Training complete!')

# Record the end time of training
end_time = time.time()
times = end_time - start_time
print(times)

data = {
    'Epoch': list(range(1, epochs + 1)),
    'D_Loss': D_Losses,
    'G_Loss': G_Losses,
    'R_Loss': R_Losses,
    'D_Real_Accuracy': D_real_accuracies,
    'D_Fake_Accuracy': D_fake_accuracies
}

# Convert training metrics to a pandas DataFrame and save it as a CSV file
df = pd.DataFrame(data)

csv_file_path = "ML/models/gan/training_metrics.csv"
df.to_csv(csv_file_path, index=False, mode='w')  # mode='w' overwrites the file

print(f"Training metrics saved to {csv_file_path}")

# Plot loss curves after training
plt.figure(figsize=(30, 8))


plt.subplot(1, 3, 1)  # Subplot 1 of 1 row and 3 columns
plt.plot(D_Losses, label="D Loss", color='red')
plt.title("Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Second subplot: G Loss
plt.subplot(1, 3, 2)  # Subplot 2 of 1 row and 3 columns
plt.plot(G_Losses, label="G Loss", color='blue')
plt.title("Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Third subplot: R Loss
plt.subplot(1, 3, 3)  # Subplot 3 of 1 row and 3 columns
plt.plot(R_Losses, label="R Loss", color='green')
plt.title("Regressor Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the figure
plt.show()

# Create a new figure for plotting accuracy curves
plt.figure(figsize=(10, 8))

# 2. Discriminator Accuracy
plt.plot(D_real_accuracies, label="Real Accuracy", color='blue')
plt.plot(D_fake_accuracies, label="Fake Accuracy", color='red')
plt.title("Discriminator Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
# plt.savefig(os.path.join(visualization_path, 'RCGAN_Acc_Curves.png'), dpi=300, bbox_inches='tight', transparent=True)
plt.show()
print("Final models have been saved.")