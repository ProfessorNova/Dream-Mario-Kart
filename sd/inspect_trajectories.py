import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 4


class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_path, sequence_length=SEQUENCE_LENGTH):
        self.trajectories = np.load(trajectory_path, allow_pickle=True)
        self.sequence_length = sequence_length
        self.data = []
        self._setup()

    def _setup(self):
        # Convert the trajectories to PyTorch tensors and normalize the images
        self.trajectories = (
            torch.tensor(self.trajectories["frames"][:1], dtype=torch.uint8) / 255.0,
            torch.tensor(self.trajectories["actions"][:1], dtype=torch.long),
        )

        # Downsample the images to half the size
        downscaled_frames = []
        for traj in range(len(self.trajectories[0])):
            downscaled_frames.append(
                torch.nn.functional.interpolate(
                    self.trajectories[0][traj], scale_factor=0.5, mode="bilinear", align_corners=False
                )
            )
        downscaled_frames = torch.stack(downscaled_frames)
        self.trajectories = (downscaled_frames, self.trajectories[1])

        # Set up the dataset
        for traj in range(len(self.trajectories[0])):
            for entry in range(len(self.trajectories[0][traj]) - self.sequence_length):
                self.data.append((
                    self.trajectories[0][traj][entry:entry + self.sequence_length],  # Previous frames
                    self.trajectories[1][traj][entry + self.sequence_length - 1],  # Action
                    self.trajectories[0][traj][entry + self.sequence_length]  # Target
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    action_names = [['NOOP'],
                    ['right'],
                    ['A'],
                    ['left']]

    # Load the dataset
    script_dir = os.path.dirname(__file__)
    dataset = TrajectoryDataset(os.path.join(script_dir, "data", "trajectories.npz"))

    for i in range(len(dataset)):
        # Display the previous frames and the target frame and action
        frames, action, target = dataset[i]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(frames[-1].permute(1, 2, 0).numpy())
        ax[0].axis("off")
        ax[0].set_title("Previous frames")
        ax[1].imshow(target.permute(1, 2, 0).numpy())
        ax[1].axis("off")
        ax[1].set_title("Target")
        plt.suptitle(f"Action: {action.item()} ({', '.join(action_names[action.item()])})")
        plt.show()


if __name__ == "__main__":
    main()
