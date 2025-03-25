import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

from lib.autoencoder import Encoder, Decoder


class FrameDataset(Dataset):
    def __init__(self, trajectory_path):
        self.trajectories = np.load(trajectory_path, allow_pickle=True)
        self.data = []
        self._setup()

    def _setup(self):
        self.data = torch.tensor(self.trajectories["frames"], dtype=torch.uint8) / 255.0
        self.data = self.data.view(-1, self.data.shape[-3], self.data.shape[-2], self.data.shape[-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def latent_regularization_loss(z):
    # Rearrange z to shape (C, B*H*W)
    B, C, H, W = z.shape
    z_flat = z.permute(1, 0, 2, 3).contiguous().view(C, -1)
    mean = z_flat.mean(dim=1)
    var = z_flat.var(dim=1)
    loss_mean = (mean ** 2).mean()  # Penalize deviation from 0
    loss_var = ((var - 1) ** 2).mean()  # Penalize deviation from 1
    return loss_mean + loss_var, mean.mean().item(), var.mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    script_dir = os.path.dirname(__file__)
    dataset = FrameDataset(os.path.join(script_dir, "data", "trajectories.npz"))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    dummy_input = next(iter(val_loader))[0]

    def train_autoencoder(encoder_model, decoder_model, train_loader, val_loader, optimizer, device,
                          latent_loss_weight=1e-3, epochs=100,
                          output_dir=os.path.join(script_dir, "output_autoencoder")):
        os.makedirs(output_dir, exist_ok=True)
        criterion = torch.nn.MSELoss()

        # Set up the gradient scaler
        scaler = torch.amp.GradScaler(str(device))
        best_val_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(1, epochs + 1):
            encoder_model.train()
            decoder_model.train()
            train_loss = 0.0
            for frames in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
                frames = frames.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast(str(device)):
                    # Get latent representation and reconstruction from the autoencoder
                    encoded = encoder_model(frames)
                    decoded = decoder_model(encoded)

                    # Calculate the loss
                    reconstruction_loss = criterion(decoded, frames)
                    regularization_loss, _, _ = latent_regularization_loss(encoded)
                    loss = reconstruction_loss + latent_loss_weight * regularization_loss

                # optimize the model
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            encoder_model.eval()
            decoder_model.eval()
            val_loss = 0.0
            latent_mean = 0.0
            latent_var = 0.0
            with torch.no_grad():
                for frames in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                    frames = frames.to(device)
                    encoded = encoder_model(frames)
                    decoded = decoder_model(encoded)
                    reconstruction_loss = criterion(decoded, frames)
                    regularization_loss, mean, std = latent_regularization_loss(encoded)
                    loss = reconstruction_loss + latent_loss_weight * regularization_loss
                    val_loss += loss.item()
                    latent_mean += mean
                    latent_var += std
            val_loss /= len(val_loader)
            latent_mean /= len(val_loader)
            latent_var /= len(val_loader)

            print(f"Epoch {epoch}/{epochs}, Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Latent Mean: {latent_mean:.6f}, Latent Var: {latent_var:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']}")

            scheduler.step(val_loss)

            # Compare images to ground truth
            if epoch % 10 == 0:
                with torch.no_grad():
                    encoder_model.eval()
                    decoder_model.eval()
                    frames = dummy_input.unsqueeze(0).to(device)
                    encoded = encoder_model(frames)
                    decoded = decoder_model(encoded)
                    comparison = torch.cat([
                        frames,
                        decoded
                    ], dim=0)
                    # Save the images
                    torchvision.utils.save_image(
                        comparison, os.path.join(output_dir, f"epoch_{epoch}.png")
                    )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving models with val loss {val_loss:.6f}")
                # Save the models
                torch.save(encoder_model, os.path.join(output_dir, "encoder.pt"))
                torch.save(decoder_model, os.path.join(output_dir, "decoder.pt"))
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter == 10:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # Init everything
    encoder = Encoder(3, 4).to(device)
    decoder = Decoder(3, 4).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    print(encoder)
    print(decoder)

    # Train the autoencoder
    train_autoencoder(encoder, decoder, train_loader, val_loader, optimizer, device, epochs=100)


if __name__ == "__main__":
    main()
