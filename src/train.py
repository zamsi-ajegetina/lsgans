"""
Training loop for LSGAN and vanilla GAN on image datasets (CelebA, LSUN).

Usage:
  python src/train.py --config configs/lsgan_celeba.yaml
  python src/train.py --config configs/dcgan_celeba.yaml
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.utils as vutils
import yaml
from tqdm import tqdm

from datasets import get_celeba_loader, get_lsun_loader, get_flat_image_loader, get_npy_loader
from models import Generator, Discriminator
from losses import (
    lsgan_loss_D, lsgan_loss_G,
    vanilla_loss_D, vanilla_loss_G,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loader(cfg: dict):
    dataset = cfg["dataset"]
    data_root = cfg["data_root"]
    bs = cfg["batch_size"]
    nw = cfg.get("num_workers", 4)

    if dataset == "celeba":
        return get_celeba_loader(data_root, image_size=64, batch_size=bs, num_workers=nw)
    elif dataset == "lsun":
        return get_lsun_loader(
            data_root,
            classes=cfg["lsun_classes"],
            image_size=64,
            batch_size=bs,
            num_workers=nw,
            max_samples=cfg.get("max_samples", None),
        )
    elif dataset == "lsun_church_kaggle":
        return get_flat_image_loader(
            data_root,
            image_size=64,
            batch_size=bs,
            num_workers=nw,
            max_samples=cfg.get("max_samples", None),
        )
    elif dataset == "lsun_church_npy":
        return get_npy_loader(
            data_root,
            batch_size=bs,
            num_workers=nw,
            max_samples=cfg.get("max_samples", None),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # directories
    exp_name = cfg["exp_name"]
    img_dir = os.path.join(cfg["output_dir"], "images", exp_name)
    ckpt_dir = os.path.join(cfg["output_dir"], "checkpoints", exp_name)
    log_path = os.path.join(cfg["output_dir"], "logs", f"{exp_name}.csv")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    loader = build_loader(cfg)

    use_bn = cfg.get("use_bn", True)
    G = Generator(latent_dim=cfg["latent_dim"], use_bn=use_bn).to(device)
    D = Discriminator(use_bn=use_bn).to(device)

    loss_type = cfg["loss"]  # "lsgan" or "vanilla"
    optimizer_type = cfg.get("optimizer", "adam")
    lr = cfg["lr"]
    beta1 = cfg.get("beta1", 0.5)

    def make_opt(params):
        if optimizer_type == "adam":
            return optim.Adam(params, lr=lr, betas=(beta1, 0.999))
        elif optimizer_type == "rmsprop":
            return optim.RMSprop(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    opt_G = make_opt(G.parameters())
    opt_D = make_opt(D.parameters())

    # fixed noise for consistent visualization across epochs
    fixed_noise = torch.randn(64, cfg["latent_dim"], device=device)

    # loss functions
    if loss_type == "lsgan":
        loss_D_fn = lsgan_loss_D
        loss_G_fn = lsgan_loss_G
    else:
        loss_D_fn = vanilla_loss_D
        loss_G_fn = vanilla_loss_G

    # logging
    log_file = open(log_path, "w")
    log_file.write("epoch,step,d_loss,g_loss,elapsed\n")

    n_epochs = cfg["n_epochs"]
    latent_dim = cfg["latent_dim"]
    global_step = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        G.train()
        D.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False)
        for batch in pbar:
            real_imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            bsz = real_imgs.size(0)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            fake_imgs = G(z).detach()
            real_out = D(real_imgs)
            fake_out = D(fake_imgs)
            d_loss = loss_D_fn(real_out, fake_out)
            d_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            z = torch.randn(bsz, latent_dim, device=device)
            fake_imgs = G(z)
            fake_out = D(fake_imgs)
            g_loss = loss_G_fn(fake_out)
            g_loss.backward()
            opt_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(d_loss=f"{d_loss.item():.4f}", g_loss=f"{g_loss.item():.4f}")

        avg_d = epoch_d_loss / n_batches
        avg_g = epoch_g_loss / n_batches
        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d} | D: {avg_d:.4f} | G: {avg_g:.4f} | {elapsed:.0f}s")
        log_file.write(f"{epoch},{global_step},{avg_d:.6f},{avg_g:.6f},{elapsed:.1f}\n")
        log_file.flush()

        # save generated image grid every epoch
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise)
        grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
        vutils.save_image(grid, os.path.join(img_dir, f"epoch_{epoch:03d}.png"))

        # save checkpoint every 5 epochs and at end
        if epoch % 5 == 0 or epoch == n_epochs:
            torch.save({
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "opt_G_state": opt_G.state_dict(),
                "opt_D_state": opt_D.state_dict(),
            }, os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt"))

    log_file.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
