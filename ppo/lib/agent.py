from typing import Tuple

import numpy as np
import torch
from torch import nn


class Agent(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super(Agent, self).__init__()
        in_channels = input_shape[-1]

        self.pooling = nn.AdaptiveAvgPool2d((84, 84))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self._get_conv_out((in_channels, 84, 84))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, num_actions)
        self.value_head = nn.Linear(512, 1)

    def _get_conv_out(self, shape: Tuple[int, int, int]) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor):
        # mix all channels of the frames bring channel dimension to the front
        # current input shape is (batch_size, num_frames, height, width, num_channels)
        # we need to convert it to (batch_size, num_frames * num_channels, height, width)
        x = x.view(x.size(0), x.size(1) * x.size(4), x.size(2), x.size(3))
        # Normalize the input
        x = x / 255.0
        # Resize the input to 84x84
        x = self.pooling(x)
        conv_out = self.conv(x)
        fc_out = self.fc(conv_out)
        return self.policy_head(fc_out), self.value_head(fc_out)
