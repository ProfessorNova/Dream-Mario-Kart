import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels=4):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_channels=4):
        super(Decoder, self).__init__()
        self.un_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid activation function to scale the pixel values between 0 and 1
        )

    def forward(self, x):
        return self.un_conv(x)
