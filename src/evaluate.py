"""
Evaluation utilities:
  - FID score computation using pytorch-fid
  - Training curve plots (D loss, G loss vs epoch)
  - Generated image grid visualization

Usage:
  # Compute FID for a trained generator
  python src/evaluate.py --config configs/lsgan_celeba.yaml \
      --ckpt experiments/checkpoints/lsgan_celeba/ckpt_epoch030.pt

  # Plot training curves for all experiments
  python src/evaluate.py --plot_curves \
      --log_dir experiments/logs --output_dir experiments/plots
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import yaml

from models import Generator


def load_generator(ckpt_path: str, latent_dim: int, use_bn: bool, device: torch.device) -> Generator:
    G = Generator(latent_dim=latent_dim, use_bn=use_bn).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    G.eval()
    return G


def save_image_grid(
    G: Generator,
    latent_dim: int,
    device: torch.device,
    out_path: str,
    n_images: int = 64,
    seed: int = 0,
) -> None:
    """Generate and save a grid of images from the trained generator."""
    torch.manual_seed(seed)
    z = torch.randn(n_images, latent_dim, device=device)
    with torch.no_grad():
        imgs = G(z)
    grid = vutils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1, 1), padding=2)
    vutils.save_image(grid, out_path)
    print(f"Saved image grid: {out_path}")


def generate_fid_samples(
    G: Generator,
    latent_dim: int,
    device: torch.device,
    out_dir: str,
    n_samples: int = 5000,
    batch_size: int = 128,
    seed: int = 0,
) -> None:
    """
    Generate n_samples images and save to out_dir as PNG files.
    These are then used as input to pytorch-fid.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    idx = 0
    G.eval()
    while idx < n_samples:
        bsz = min(batch_size, n_samples - idx)
        z = torch.randn(bsz, latent_dim, device=device)
        with torch.no_grad():
            imgs = G(z)
        # unnormalize from [-1,1] to [0,1]
        imgs = (imgs + 1.0) / 2.0
        imgs = imgs.clamp(0, 1)
        for img in imgs:
            vutils.save_image(img, os.path.join(out_dir, f"{idx:05d}.png"))
            idx += 1
    print(f"Saved {n_samples} images to {out_dir}")


def read_log(log_path: str) -> dict:
    """Parse CSV training log into arrays."""
    data = {"epoch": [], "d_loss": [], "g_loss": []}
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["d_loss"].append(float(row["d_loss"]))
            data["g_loss"].append(float(row["g_loss"]))
    return data


def plot_training_curves(log_dir: str, output_dir: str) -> None:
    """
    Plot discriminator and generator loss curves for all experiments
    found in log_dir, overlaid on the same figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = plt.cm.tab10.colors
    for i, fname in enumerate(sorted(log_files)):
        name = fname.replace(".csv", "")
        data = read_log(os.path.join(log_dir, fname))
        c = colors[i % len(colors)]
        axes[0].plot(data["epoch"], data["d_loss"], label=name, color=c)
        axes[1].plot(data["epoch"], data["g_loss"], label=name, color=c)

    axes[0].set_title("Discriminator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Generator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main(args):
    if args.plot_curves:
        plot_training_curves(args.log_dir, args.output_dir)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = load_generator(
        args.ckpt,
        cfg["latent_dim"],
        cfg.get("use_bn", True),
        device,
    )

    exp_name = cfg["exp_name"]
    os.makedirs(args.output_dir, exist_ok=True)

    # save a visual grid
    grid_path = os.path.join(args.output_dir, f"{exp_name}_grid.png")
    save_image_grid(G, cfg["latent_dim"], device, grid_path)

    # generate samples for FID
    fid_dir = os.path.join(args.output_dir, f"{exp_name}_fid_samples")
    generate_fid_samples(G, cfg["latent_dim"], device, fid_dir, n_samples=args.fid_samples)

    print(f"\nTo compute FID, run:")
    print(f"  python -m pytorch_fid {fid_dir} <path_to_real_image_stats_or_folder>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output_dir", default="experiments/plots")
    parser.add_argument("--log_dir", default="experiments/logs")
    parser.add_argument("--fid_samples", type=int, default=5000)
    parser.add_argument("--plot_curves", action="store_true")
    args = parser.parse_args()
    main(args)
