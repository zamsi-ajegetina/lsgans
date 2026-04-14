"""
Generator and Discriminator architectures for LSGAN.

Adapted from paper Figure 3 for 64x64 output instead of 112x112.
Generator uses ReLU + BN; Discriminator uses LeakyReLU + BN (except first layer).
Both LSGAN and vanilla GAN share the same architecture; the only difference
is whether the discriminator's final output passes through sigmoid (vanilla)
or not (LSGAN, which applies least squares loss directly on the raw score).
"""

import torch
import torch.nn as nn


def _weights_init(m: nn.Module) -> None:
    """Initialize conv and batchnorm weights as in DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator network. Maps z (latent_dim,) -> (3, 64, 64) image.

    Architecture (adapted from paper Fig. 3a for 64x64):
      z -> fc -> reshape 4x4x1024 -> BN
      -> deconv 4x4/512/stride2 -> BN -> ReLU     => 8x8
      -> deconv 4x4/256/stride2 -> BN -> ReLU     => 16x16
      -> deconv 4x4/128/stride2 -> BN -> ReLU     => 32x32
      -> deconv 4x4/3/stride2   -> Tanh           => 64x64
    """

    def __init__(self, latent_dim: int = 1024, use_bn: bool = True) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        def deconv_block(in_ch, out_ch, bn=True):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 1024, bias=False),
            nn.BatchNorm1d(4 * 4 * 1024) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.conv_blocks = nn.Sequential(
            *deconv_block(1024, 512, bn=use_bn),
            *deconv_block(512, 256, bn=use_bn),
            *deconv_block(256, 128, bn=use_bn),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

        self.apply(_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 1024, 4, 4)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """
    Discriminator network. Maps (3, 64, 64) -> scalar score.

    Architecture (adapted from paper Fig. 3b for 64x64):
      -> conv 4x4/64/stride2   -> LeakyReLU       => 32x32
      -> conv 4x4/128/stride2  -> BN -> LeakyReLU => 16x16
      -> conv 4x4/256/stride2  -> BN -> LeakyReLU => 8x8
      -> conv 4x4/512/stride2  -> BN -> LeakyReLU => 4x4
      -> fc -> 1

    For LSGAN: no sigmoid on output (raw score fed to MSE loss).
    For vanilla GAN: sigmoid applied in the loss function via
      binary_cross_entropy_with_logits, so output is also raw logit.
    """

    def __init__(self, use_bn: bool = True) -> None:
        super().__init__()
        negative_slope = 0.2

        def conv_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            return layers

        self.conv_blocks = nn.Sequential(
            # first layer: no BN per DCGAN convention
            *conv_block(3, 64, bn=False),
            *conv_block(64, 128, bn=use_bn),
            *conv_block(128, 256, bn=use_bn),
            *conv_block(256, 512, bn=use_bn),
        )

        self.fc = nn.Linear(512 * 4 * 4, 1)

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_blocks(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)
