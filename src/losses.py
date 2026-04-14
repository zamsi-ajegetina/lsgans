"""
Loss functions for LSGAN and vanilla GAN.

LSGAN uses Equation 9 from the paper (0-1 coding scheme, c=b=1):
  L_D = 0.5 * E[(D(x) - 1)^2] + 0.5 * E[(D(G(z)))^2]
  L_G = 0.5 * E[(D(G(z)) - 1)^2]

Vanilla GAN uses sigmoid cross-entropy:
  L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
  L_G = -E[log D(G(z))]
"""

import torch
import torch.nn.functional as F


def lsgan_loss_D(real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
    """Discriminator loss for LSGAN (Eq. 9)."""
    real_loss = 0.5 * torch.mean((real_output - 1.0) ** 2)
    fake_loss = 0.5 * torch.mean(fake_output ** 2)
    return real_loss + fake_loss


def lsgan_loss_G(fake_output: torch.Tensor) -> torch.Tensor:
    """Generator loss for LSGAN (Eq. 9)."""
    return 0.5 * torch.mean((fake_output - 1.0) ** 2)


def vanilla_loss_D(real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
    """Discriminator loss for vanilla GAN (binary cross-entropy)."""
    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)
    real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
    return real_loss + fake_loss


def vanilla_loss_G(fake_output: torch.Tensor) -> torch.Tensor:
    """Generator loss for vanilla GAN (non-saturating)."""
    real_labels = torch.ones_like(fake_output)
    return F.binary_cross_entropy_with_logits(fake_output, real_labels)
