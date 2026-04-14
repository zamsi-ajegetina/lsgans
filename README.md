# LSGAN Replication — ICS 555 Paper Replication Assignment

Replication of **"Least Squares Generative Adversarial Networks"** (Mao et al., ICCV 2017 / arXiv 1611.04076).

## Scope

The original paper trains on LSUN at 112×112 (up to 3M images). This replication uses:
- **CelebA at 64×64** as the primary dataset (publicly available, ~200k images)
- **8-Gaussian mixture** (2D synthetic) for the exact stability comparison from Section 4.3
- Architecture adapted from paper Figure 3 (same design principles, scaled for 64×64)

## Setup

```bash
pip install -r requirements.txt
```

## Data

CelebA is downloaded automatically by torchvision on first run:
```bash
# Data will be saved to data/celeba/
python src/train.py --config configs/lsgan_celeba.yaml
```

## Experiments

### 1. Gaussian Mixture Stability (replicates paper Figure 8)
```bash
python src/gaussian_exp.py --output_dir experiments/plots --seed 42
# Output: experiments/plots/gaussian_mixture_stability.png
```

### 2. LSGAN vs Vanilla GAN on CelebA
```bash
# Train LSGAN
python src/train.py --config configs/lsgan_celeba.yaml

# Train vanilla DCGAN baseline
python src/train.py --config configs/dcgan_celeba.yaml
```

### 3. Stability Ablation (no BN)
```bash
# LSGAN without BatchNorm
python src/train.py --config configs/lsgan_celeba_nobn.yaml

# Vanilla GAN without BatchNorm
python src/train.py --config configs/dcgan_celeba_nobn.yaml
```

### 4. Evaluation (FID + image grids + training curves)
```bash
# Generate image grid and FID samples
python src/evaluate.py \
    --config configs/lsgan_celeba.yaml \
    --ckpt experiments/checkpoints/lsgan_celeba/ckpt_epoch030.pt \
    --output_dir experiments/plots

# Compute FID (requires real image stats)
python -m pytorch_fid \
    experiments/plots/lsgan_celeba_fid_samples \
    data/celeba/img_align_celeba \
    --device cuda

# Plot training curves for all experiments
python src/evaluate.py --plot_curves \
    --log_dir experiments/logs \
    --output_dir experiments/plots
```

## Reproducing Report Figures

| Figure | Command | Output |
|--------|---------|--------|
| Gaussian stability (Fig. 8 equivalent) | `python src/gaussian_exp.py` | `experiments/plots/gaussian_mixture_stability.png` |
| Generated image grids | `python src/evaluate.py --config ... --ckpt ...` | `experiments/plots/*_grid.png` |
| Training loss curves | `python src/evaluate.py --plot_curves` | `experiments/plots/training_curves.png` |

## Project Structure

```
lsgan/
├── requirements.txt
├── README.md
├── src/
│   ├── datasets.py       # CelebA, LSUN, Gaussian mixture loaders
│   ├── models.py         # Generator and Discriminator
│   ├── losses.py         # LSGAN (Eq.9) and vanilla GAN losses
│   ├── train.py          # Training loop
│   ├── evaluate.py       # FID, image grids, training curves
│   └── gaussian_exp.py   # 8-Gaussian stability experiment
├── configs/              # YAML configs for each experiment
└── experiments/          # Outputs (checkpoints, images, logs, plots)
```

## Reproducibility

All experiments fix random seeds via `set_seed()` in `train.py` and `gaussian_exp.py`.
Running the same command twice produces identical results.

## References

Mao et al., "Least Squares Generative Adversarial Networks," ICCV 2017.
arXiv: https://arxiv.org/abs/1611.04076
# lsgans
