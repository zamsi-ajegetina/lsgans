"""
Data loading utilities.

Supported datasets:
  - CelebA (64x64, downloaded via torchvision)
  - LSUN Church (subset, downloaded via torchvision)
  - LSUN Church from Kaggle (flat image directory, e.g. kagglehub download)
  - Gaussian mixture (synthetic 2D, for stability experiment)
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as T


def get_celeba_loader(
    data_root: str,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "train",
) -> DataLoader:
    """
    Returns a DataLoader for CelebA at the given image_size.
    Images are normalized to [-1, 1] to match the generator's Tanh output.
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CelebA(
        root=data_root,
        split=split,
        transform=transform,
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_lsun_loader(
    data_root: str,
    classes: list,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: int = None,
) -> DataLoader:
    """
    Returns a DataLoader for LSUN scene categories.
    classes: list of strings, e.g. ["church_outdoor_train"]
    max_samples: if set, use only the first N samples (for subset experiments).
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.LSUN(
        root=data_root,
        classes=classes,
        transform=transform,
    )
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class NpyImageDataset(Dataset):
    """
    Loads images from a .npy file of shape (N, H, W, 3) uint8.
    Normalizes to [-1, 1] to match generator Tanh output.
    Used for the pre-processed LSUN Church 64x64 numpy array.
    """

    def __init__(self, npy_path: str, transform=None):
        self.data = np.load(npy_path)  # (N, H, W, 3) uint8
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]  # (H, W, 3) uint8
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0,1]
        img = img * 2.0 - 1.0  # normalize to [-1, 1]
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label


def get_npy_loader(
    npy_path: str,
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples: int = None,
) -> DataLoader:
    """
    DataLoader for a pre-processed .npy image array (N, H, W, 3) uint8.
    Images are already 64x64 — no resize needed.
    """
    dataset = NpyImageDataset(npy_path)
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


class FlatImageDataset(Dataset):
    """
    Loads all images from a flat directory (no class subdirs).
    Used for Kaggle-downloaded LSUN Church which is a folder of JPEGs.
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.paths = [
            p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in self.EXTENSIONS
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label to match (image, label) convention


def get_flat_image_loader(
    data_root: str,
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples: int = None,
) -> DataLoader:
    """
    DataLoader for a flat directory of images (e.g. Kaggle LSUN Church).
    Images are center-cropped, resized, and normalized to [-1, 1].
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = FlatImageDataset(data_root, transform=transform)
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def sample_gaussian_mixture(
    n_samples: int,
    n_modes: int = 8,
    radius: float = 2.0,
    std: float = 0.02,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sample from a 2D mixture of n_modes Gaussians arranged in a circle.
    Used for the stability comparison experiment (Section 4.3 of paper).
    """
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    samples_per_mode = n_samples // n_modes
    samples = []
    for center in centers:
        noise = np.random.randn(samples_per_mode, 2) * std
        samples.append(center + noise)
    samples = np.concatenate(samples, axis=0)
    np.random.shuffle(samples)
    return torch.tensor(samples, dtype=torch.float32, device=device)


def get_gaussian_mixture_loader(
    n_samples: int = 10000,
    batch_size: int = 512,
    n_modes: int = 8,
    radius: float = 2.0,
    std: float = 0.02,
) -> DataLoader:
    """Returns a DataLoader for the 2D Gaussian mixture dataset."""
    data = sample_gaussian_mixture(n_samples, n_modes, radius, std)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
