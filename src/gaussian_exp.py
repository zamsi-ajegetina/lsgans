"""
Stability experiment on 2D Gaussian mixture (Section 4.3 of paper).

Trains both LSGAN and vanilla GAN on a mixture of 8 Gaussians arranged
in a circle, then visualizes kernel density estimates at steps 0, 5k, 15k, 25k, 40k.

Usage:
  python src/gaussian_exp.py --output_dir experiments/plots --seed 42
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import gaussian_kde

from datasets import sample_gaussian_mixture
from losses import lsgan_loss_D, lsgan_loss_G, vanilla_loss_D, vanilla_loss_G


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Simple 3-layer fully-connected network (256 hidden units) for 2D data."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def plot_kde(
    ax,
    samples: np.ndarray,
    title: str,
    xlim=(-3.5, 3.5),
    ylim=(-3.5, 3.5),
    n_grid: int = 200,
) -> None:
    """Plot kernel density estimate on the given axes."""
    kde = gaussian_kde(samples.T, bw_method=0.1)
    xx, yy = np.meshgrid(
        np.linspace(*xlim, n_grid),
        np.linspace(*ylim, n_grid),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(positions).reshape(n_grid, n_grid)
    ax.contourf(xx, yy, z, levels=20, cmap="Greens")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def run_experiment(
    loss_type: str,
    total_steps: int,
    snapshot_steps: list,
    batch_size: int,
    latent_dim: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> dict:
    """
    Train a GAN on the 8-Gaussian mixture and collect generated sample
    snapshots at the specified training steps.

    Returns dict: {step -> np.ndarray of shape (N, 2)}
    """
    set_seed(seed)

    G = MLP(latent_dim, 2).to(device)
    D = MLP(2, 1).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    if loss_type == "lsgan":
        loss_D_fn, loss_G_fn = lsgan_loss_D, lsgan_loss_G
    else:
        loss_D_fn, loss_G_fn = vanilla_loss_D, vanilla_loss_G

    snapshots = {}
    step = 0

    while step < total_steps:
        real = sample_gaussian_mixture(batch_size, device=device)

        # train D
        opt_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z).detach()
        d_loss = loss_D_fn(D(real), D(fake))
        d_loss.backward()
        opt_D.step()

        # train G
        opt_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)
        g_loss = loss_G_fn(D(fake))
        g_loss.backward()
        opt_G.step()

        step += 1

        if step in snapshot_steps:
            G.eval()
            with torch.no_grad():
                z = torch.randn(2048, latent_dim, device=device)
                gen_samples = G(z).cpu().numpy()
            snapshots[step] = gen_samples
            G.train()
            print(f"  [{loss_type}] step {step}/{total_steps} snapshot saved")

    return snapshots


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    snapshot_steps = [5000, 15000, 25000, 40000]
    total_steps = 40000
    batch_size = 512
    latent_dim = 256
    lr = 1e-4

    # ground truth target distribution
    target_samples = sample_gaussian_mixture(4096).numpy()

    print("Training LSGAN on Gaussian mixture...")
    lsgan_snapshots = run_experiment(
        "lsgan", total_steps, snapshot_steps, batch_size, latent_dim, lr, device, args.seed
    )

    print("Training vanilla GAN on Gaussian mixture...")
    vanilla_snapshots = run_experiment(
        "vanilla", total_steps, snapshot_steps, batch_size, latent_dim, lr, device, args.seed + 1
    )

    # --- Plot: replicate paper Figure 8 layout ---
    # Rows: [LSGANs, Regular GANs]
    # Columns: [Step 5k, 15k, 25k, 40k, Target]
    col_labels = ["Step 5k", "Step 15k", "Step 25k", "Step 40k", "Target"]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))

    for col_idx, step in enumerate(snapshot_steps):
        plot_kde(axes[0, col_idx], lsgan_snapshots[step], col_labels[col_idx])
        plot_kde(axes[1, col_idx], vanilla_snapshots[step], col_labels[col_idx])

    # target column
    plot_kde(axes[0, 4], target_samples, "Target")
    plot_kde(axes[1, 4], target_samples, "Target")

    axes[0, 0].set_ylabel("LSGANs", fontsize=10)
    axes[1, 0].set_ylabel("Regular GANs", fontsize=10)

    plt.suptitle(
        "Gaussian mixture: kernel density estimation (replicating Fig. 8)",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "gaussian_mixture_stability.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
