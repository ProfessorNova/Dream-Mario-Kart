import os
import time

import numpy as np
import pygame
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset
from tqdm import tqdm

from lib.sd_functions import generate
from lib.unet import UNet

SEQUENCE_LENGTH = 4


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
            torch.tensor(self.trajectories["frames"][:1], dtype=torch.uint8) / 255.0,
            torch.tensor(self.trajectories["actions"][:1], dtype=torch.long),
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

    script_dir = os.path.dirname(__file__)
    latent_shape = (4, 28, 64)
    image_shape = (3, 112, 256)

    decoder = torch.load(os.path.join(script_dir, "pretrained", "decoder.pt"), weights_only=False).to(device)
    decoder.eval()

    model = UNet(
        in_shape=latent_shape,
        out_shape=latent_shape,
        features=[64, 128, 256, 512],
        embedding_dim=64,
        action_vocab_size=6,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    print(model)

    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
        device=device
    )
    ema_model.load_state_dict(
        torch.load(os.path.join(script_dir, "pretrained", "model.pt"), map_location=device, weights_only=True))
    print("Model loaded successfully!")

    # Load the initial sequence
    initial_sequence = np.load(os.path.join(script_dir, "pretrained", "initial_sequence.npz"))
    frame_buffer = torch.tensor(initial_sequence["frames"], dtype=torch.float32).to(device)
    action_buffer = torch.tensor(initial_sequence["actions"], dtype=torch.long).to(device)

    pygame.init()
    display_width = image_shape[2] * 4
    display_height = image_shape[1] * 4
    screen = pygame.display.set_mode((display_width, display_height))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
            action = 4
        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
            action = 5
        elif keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3
        else:
            action = 0

        # Roll the action buffer
        action_buffer = torch.cat([action_buffer[:, 1:], torch.tensor([action], device=device).unsqueeze(0)], dim=1)

        with torch.no_grad():
            start_time = time.time()
            generated_frame = generate(ema_model,
                                       1,
                                       7,
                                       latent_shape,
                                       frame_buffer,
                                       action_buffer,
                                       device)
            print(f"Frame time: {time.time() - start_time:.4f}s (max. fps: {1 / (time.time() - start_time):.2f})",
                  end="\r")

            # Add the generated frame to the buffer
            frame_buffer = torch.cat([frame_buffer[:, 1:], generated_frame.unsqueeze(0)], dim=1)
            # Extract the frame from the buffer for display
            frame = (decoder(frame_buffer[0, -1]).cpu().numpy().transpose(2, 1, 0) * 255).astype("uint8")

        # Scale the frame up for display
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (display_width, display_height))
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(8)

    pygame.quit()


if __name__ == "__main__":
    main()
