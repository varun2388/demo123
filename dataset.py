"""
Dataset loading and preprocessing utilities for Vision Transformer training.

Uses CIFAR-10 by default — a widely-used benchmark with 10 classes of 32x32 RGB images.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 channel-wise mean and std for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transforms():
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transforms():
    """Test/inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_dataloaders(data_dir="./data", batch_size=128, num_workers=2):
    """Create CIFAR-10 training and test data loaders.

    Args:
        data_dir: Directory to download/load the dataset.
        batch_size: Batch size for training and evaluation.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_train_transforms()
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
