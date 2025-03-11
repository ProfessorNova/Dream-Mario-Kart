import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FILE_DIR = os.path.dirname(__file__)


def sinusoidal_embedding(x, embedding_dim=32):
    frequencies = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            embedding_dim // 2,
            device=x.device,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.concat(
        [torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], dim=-1
    )
    return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, action_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # If channel dimensions differ, adapt with 1x1 convolution
        self.channel_adaptation = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        self.norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()

        # Projection for action tokens to match out_channels if needed
        self.action_proj = nn.Linear(action_emb_dim, out_channels) \
            if action_emb_dim != out_channels else nn.Identity()

        # Cross-Attention: image features are queries; action tokens (projected) serve as key and value
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, action_token):
        residual = x if self.in_channels == self.out_channels else self.channel_adaptation(x)
        out = self.norm(residual)
        out = self.conv1(out)
        out = self.act(out)

        # Flatten spatial dimensions for attention
        B, C, H, W = out.shape
        out_flat = out.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

        # Project action tokens to match feature dimensions
        action_proj = self.action_proj(action_token)  # (B, L, out_channels)

        # Apply cross-attention: Query = flattened image features, Key/Value = action tokens
        attn_out, _ = self.cross_attn(
            query=out_flat,
            key=action_proj,
            value=action_proj
        )
        # Residual connection around attention
        out_flat = self.ln(out_flat + attn_out)

        # Reshape back to 2D spatial format
        out = out_flat.transpose(1, 2).view(B, C, H, W)
        out = self.conv2(out)
        return out + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, action_emb_dim, block_depth=2):
        super().__init__()
        self.block_depth = block_depth

        # First ResidualBlock adjusts in_channels to out_channels
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels, out_channels, action_emb_dim)
        ])
        # Other ResidualBlocks keep the same number of channels
        for _ in range(block_depth - 1):
            self.residual_blocks.append(ResidualBlock(out_channels, out_channels, action_emb_dim))
        self.pool = nn.AvgPool2d(2)

    def forward(self, x_tupel, action_token):
        x, skips = x_tupel
        for block in self.residual_blocks:
            x = block(x, action_token)  # Apply the block with the action token
            skips.append(x)  # Save the skip connection
        x = self.pool(x)  # Downsample the output
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, action_emb_dim, block_depth=2):
        super().__init__()
        self.block_depth = block_depth
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # First ResidualBlock needs to take the upsampled input concatenated with the skip connection
        # The skip connection from the corresponding down block has the same number of channels as the output
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels + out_channels, out_channels, action_emb_dim)
        ])
        # Other ResidualBlocks needs to take the output of the previous block concatenated with the skip connection
        for _ in range(block_depth - 1):
            self.residual_blocks.append(ResidualBlock(out_channels + out_channels, out_channels, action_emb_dim))

    def forward(self, x_tupel, action_token):
        x, skips = x_tupel
        x = self.upsample(x)  # Upsample the input
        for block in self.residual_blocks:
            skip = skips.pop()  # Last skip connection

            # Padding if the spatial dimensions do not match
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - x.shape[-2]
                diff_x = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                              diff_y // 2, diff_y - diff_y // 2])

            x = torch.cat([x, skip], dim=1)  # Concatenate the skip connection
            x = block(x, action_token)  # Apply the block with the action token
        return x


class UNet(nn.Module):
    def __init__(self, in_shape, out_shape, features=None, embedding_dim=64, action_vocab_size=4, sequence_length=4):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.features = features
        self.embedding_dim = embedding_dim
        self.action_vocab_size = action_vocab_size
        self.sequence_length = sequence_length

        # Embeddings
        self.embedding_upsample = nn.UpsamplingBilinear2d(size=in_shape[1:])
        self.in_conv = nn.Conv2d(in_shape[0], features[0], kernel_size=1)

        # Action embedding
        self.action_embedding = nn.Embedding(action_vocab_size, embedding_dim)

        self.down_blocks = nn.ModuleList([
            # First Block is concatenated with the sinusoidal embedding for variance -> (+ embedding_dim)
            # The previous frames are concatenated as well -> (+ in_shape[0] * sequence_length)
            DownBlock(features[0] + embedding_dim + in_shape[0] * sequence_length, features[0], embedding_dim)
        ])
        for i in range(len(features) - 2):  # -2 because the first block is already added
            self.down_blocks.append(DownBlock(features[i], features[i + 1], embedding_dim))

        self.bottleneck = nn.ModuleList([
            # First ResidualBlock adjusts the number of channels of the last down block
            ResidualBlock(features[-2], features[-1], embedding_dim),
            ResidualBlock(features[-1], features[-1], embedding_dim)
        ])

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(features) - 1)):
            # First up block will take the bottleneck output
            self.up_blocks.append(UpBlock(features[i + 1], features[i], embedding_dim))

        self.out_conv = nn.Conv2d(features[0], out_shape[0], kernel_size=1)

    def forward(self, x, t, previous_frames, previous_actions):
        # x is the next frame in latent space
        # t are noise variances
        # previous_frames are the previous frames in latent space
        # action is the action at the current frame

        # Create the sinusoidal embedding
        t_embedding = sinusoidal_embedding(t, self.embedding_dim).to(x.device)
        t_embedding = t_embedding.view(-1, self.embedding_dim, 1, 1)
        t_embedding = self.embedding_upsample(t_embedding)

        # Embed the previous actions
        action_tokens = self.action_embedding(previous_actions)  # (B, sequence_length, embedding_dim)
        # Add position encoding to the action tokens
        action_pos_embedding = sinusoidal_embedding(torch.arange(self.sequence_length).unsqueeze(1),
                                                    self.embedding_dim).to(x.device)
        action_pos_embedding = action_pos_embedding.unsqueeze(0)
        action_tokens = action_tokens + action_pos_embedding

        # Concatenate the previous frames
        previous_frames = previous_frames.view(previous_frames.shape[0], -1, previous_frames.shape[-2],
                                               previous_frames.shape[-1])  # (B, in_shape[0] * sequence_length, H, W)

        # Bring the input to the right shape and concatenate the embedding
        x = self.in_conv(x)
        x = torch.cat([x, t_embedding, previous_frames], dim=1)

        # List to store the skip connections
        skips = []

        # Downward pass
        for block in self.down_blocks:
            x = block([x, skips], action_tokens)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x, action_tokens)

        # Upward pass
        for block in self.up_blocks:
            x = block([x, skips], action_tokens)

        # Bring the output to the image shape
        x = self.out_conv(x)

        return x
