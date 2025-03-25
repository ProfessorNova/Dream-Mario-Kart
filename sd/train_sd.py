import os

import numpy as np
import torch
import torchvision
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.sd_functions import cosine_diffusion_schedule, denoise, generate
from lib.unet import UNet

SEQUENCE_LENGTH = 4
NUM_ACTIONS = 6


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_path, encoder, sequence_length, device):
        self.trajectories = np.load(trajectory_path, allow_pickle=True)
        self.encoder = encoder
        self.sequence_length = sequence_length
        self.device = device
        self.data = []
        self._setup()

    def _setup(self):
        # Convert the trajectories to PyTorch tensors and normalize the images
        self.trajectories = (
            torch.tensor(self.trajectories["frames"], dtype=torch.uint8) / 255.0,
            torch.tensor(self.trajectories["actions"], dtype=torch.long),
        )

        # Encode the images
        with torch.no_grad():
            encoded_frames = []
            for traj in tqdm(range(len(self.trajectories[0])), desc="Encoding Frames"):
                encoded_frames.append(self.encoder(self.trajectories[0][traj].to(self.device)).cpu())
            encoded_frames = torch.stack(encoded_frames)
        self.trajectories = (encoded_frames, self.trajectories[1])

        # Set up the dataset
        for traj in range(len(self.trajectories[0])):
            for entry in range(len(self.trajectories[0][traj]) - self.sequence_length):
                self.data.append((
                    self.trajectories[0][traj][entry:entry + self.sequence_length],  # Previous frames
                    self.trajectories[1][traj][entry:entry + self.sequence_length],  # Previous actions
                    self.trajectories[0][traj][entry + self.sequence_length]  # Target
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the autoencoder
    encoder = torch.load("output_autoencoder/encoder.pt", weights_only=False).to(device)
    decoder = torch.load("output_autoencoder/decoder.pt", weights_only=False).to(device)
    encoder.eval()
    decoder.eval()

    # Load the dataset
    script_dir = os.path.dirname(__file__)
    dataset = TrajectoryDataset(os.path.join(script_dir, "data", "trajectories.npz"), encoder, SEQUENCE_LENGTH, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    dummy_input = next(iter(loader))[2][0]
    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Number of Samples: {len(dataset)}")

    def train_diffusion_model(model, ema_model, loader, optimizer, decoder, device, epochs=50, max_noise_level=0.15,
                              output_dir=os.path.join(script_dir, "output_sd")):
        os.makedirs(output_dir, exist_ok=True)

        # Save an initial sequence for later inference
        initial_sequence_frame_buffer = dataset[0][0].unsqueeze(0).cpu().numpy()
        initial_sequence_action_buffer = dataset[0][1].unsqueeze(0).cpu().numpy()
        np.savez(os.path.join(output_dir, "initial_sequence.npz"), frames=initial_sequence_frame_buffer,
                 actions=initial_sequence_action_buffer)

        criterion = torch.nn.MSELoss()

        # Use gradient scaling to prevent underflow
        scaler = torch.amp.GradScaler(str(device))

        best_loss = float("inf")
        early_stopping_counter = 0

        for epoch in range(1, epochs + 1):
            avg_loss = 0.0
            model.train()
            for prev_frames, prev_actions, targets in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
                prev_frames = prev_frames.to(device, non_blocking=True)
                prev_actions = prev_actions.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                noises = torch.randn(targets.shape, device=device)

                # Sample a random timestep for each image in range [0, 1]
                diffusion_times = torch.rand(targets.shape[0], 1, 1, 1, device=device)
                noise_rates, signal_rates = cosine_diffusion_schedule(diffusion_times, device)
                # Mix the images with the noise
                noisy_images = signal_rates * targets + noise_rates * noises

                # Apply noise augmentation to the previous frames
                noise_level = torch.rand(prev_frames.size(0), 1, 1, 1, 1, device=device) * max_noise_level
                noise_rates_aug, signal_rates_aug = cosine_diffusion_schedule(noise_level, device)
                noise_aug = torch.randn_like(prev_frames)
                prev_frames = signal_rates_aug * prev_frames + noise_rates_aug * noise_aug

                optimizer.zero_grad()
                # Forward Pass mit autocast
                with torch.amp.autocast(str(device)):
                    predicted_noises, _ = denoise(model, noisy_images, noise_rates, signal_rates, prev_frames,
                                                  prev_actions)
                    loss = criterion(noises, predicted_noises)

                # Backward Pass und Optimierung mit GradScaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # Aktualisieren Sie die EMA-Parameter
                ema_model.update_parameters(model)

                avg_loss += loss.item()

            avg_loss /= len(loader)
            print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")
            scheduler.step(avg_loss)

            if epoch % 10 == 0:
                # Sample images and save them as a grid
                with torch.no_grad():
                    previous_frames, _, _ = next(iter(loader))
                    previous_frames = previous_frames[0:NUM_ACTIONS].to(device)
                    # Generate images for all actions
                    prev_actions = torch.arange(NUM_ACTIONS, dtype=torch.int64, device=device)
                    # Repeat the previous actions for each image
                    prev_actions = prev_actions.repeat(SEQUENCE_LENGTH, 1).T
                    generated_images = generate(ema_model,
                                                NUM_ACTIONS,
                                                20,
                                                dummy_input.shape,
                                                previous_frames,
                                                prev_actions,
                                                device)
                    generated_images = torch.cat([
                        decoder(previous_frames[:, -1]),
                        decoder(generated_images)
                    ], dim=0)
                    torchvision.utils.save_image(
                        generated_images, f"{output_dir}/epoch_{epoch}.png", nrow=NUM_ACTIONS
                    )

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"Saving model with new best loss: {best_loss:.4f}")
                torch.save(ema_model.state_dict(), f"{output_dir}/best.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 20:
                    print("Early stopping")
                    break

            # Save the last model
            torch.save(ema_model.state_dict(), f"{output_dir}/last.pt")

    # Initialize models, optimizer, and EMA
    model = UNet(
        in_shape=dummy_input.shape,
        out_shape=dummy_input.shape,
        features=[64, 128, 256, 512],
        embedding_dim=64,
        action_vocab_size=NUM_ACTIONS,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    print(model)

    # Use EMA for model so that the model is more robust to noise
    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
        device=device
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # Start training
    train_diffusion_model(model, ema_model, loader, optimizer, decoder, device, epochs=300)


if __name__ == "__main__":
    main()
