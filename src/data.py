"""
src/data.py
Dataset loading and Dirichlet Non-IID partitioning.
Supports: MNIST, FMNIST, CIFAR-10, CIFAR-100

Key fix: DATA_ROOT is always the absolute path  <repo_root>/data/
so that the server subprocess and all client subprocesses find the
same files regardless of their working directory.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple

# Absolute path to data directory — always <repo_root>/data/
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
DATA_ROOT  = os.path.join(_REPO_ROOT, "data")
os.makedirs(DATA_ROOT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

TRANSFORMS = {
    "MNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "FMNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]),
    "CIFAR10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ]),
    "CIFAR100": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
}

TEST_TRANSFORMS = {
    "MNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "FMNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]),
    "CIFAR10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ]),
    "CIFAR100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader  (download=True but data already exists from pre-download)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str):
    """Load (train_dataset, test_dataset) from DATA_ROOT."""
    ds = dataset_name.upper()

    if ds == "MNIST":
        train = torchvision.datasets.MNIST(
            DATA_ROOT, train=True,  download=True, transform=TRANSFORMS["MNIST"])
        test  = torchvision.datasets.MNIST(
            DATA_ROOT, train=False, download=True, transform=TEST_TRANSFORMS["MNIST"])

    elif ds == "FMNIST":
        train = torchvision.datasets.FashionMNIST(
            DATA_ROOT, train=True,  download=True, transform=TRANSFORMS["FMNIST"])
        test  = torchvision.datasets.FashionMNIST(
            DATA_ROOT, train=False, download=True, transform=TEST_TRANSFORMS["FMNIST"])

    elif ds == "CIFAR10":
        train = torchvision.datasets.CIFAR10(
            DATA_ROOT, train=True,  download=True, transform=TRANSFORMS["CIFAR10"])
        test  = torchvision.datasets.CIFAR10(
            DATA_ROOT, train=False, download=True, transform=TEST_TRANSFORMS["CIFAR10"])

    elif ds == "CIFAR100":
        train = torchvision.datasets.CIFAR100(
            DATA_ROOT, train=True,  download=True, transform=TRANSFORMS["CIFAR100"])
        test  = torchvision.datasets.CIFAR100(
            DATA_ROOT, train=False, download=True, transform=TEST_TRANSFORMS["CIFAR100"])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Choose from MNIST, FMNIST, CIFAR10, CIFAR100.")

    return train, test


# ──────────────────────────────────────────────────────────────────────────────
# Dirichlet Non-IID partitioning
# ──────────────────────────────────────────────────────────────────────────────

def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset indices via Dirichlet distribution.
    Smaller alpha → more non-IID.
    Returns list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array([
        dataset.targets[i] if isinstance(dataset.targets, list)
        else int(dataset.targets[i])
        for i in range(len(dataset))
    ])
    num_classes = len(np.unique(targets))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(cls_idx)).astype(int)
        diff = len(cls_idx) - proportions.sum()
        proportions[: abs(diff)] += int(np.sign(diff))
        splits = np.split(cls_idx, np.cumsum(proportions)[:-1])
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    return client_indices


def iid_partition(dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """IID partition — shuffle and split equally."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return [chunk.tolist() for chunk in np.array_split(indices, num_clients)]


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_client_loaders(
    dataset,
    client_indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    subset = Subset(dataset, client_indices)
    # drop_last=True: drops the final incomplete batch so BatchNorm never
    # receives a single-sample batch (which causes "Expected more than 1
    # value per channel" ValueError with ResNet18 / AlexNet).
    # Also cap batch_size to half the client's data so at least 2 batches exist.
    effective_bs = min(batch_size, max(2, len(client_indices) // 2))
    return DataLoader(subset, batch_size=effective_bs, shuffle=shuffle,
                      num_workers=0, pin_memory=False, drop_last=True)


def get_test_loader(test_dataset, batch_size: int = 128) -> DataLoader:
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ──────────────────────────────────────────────────────────────────────────────
# Build all client partitions from config
# ──────────────────────────────────────────────────────────────────────────────

def build_partitions(cfg: dict):
    """
    Returns (train_dataset, test_dataset, partitions).
    partitions[i] = list of indices for client i.
    """
    train_ds, test_ds = load_dataset(cfg["data"]["dataset"])

    alpha       = cfg["data"]["dirichlet_alpha"]
    num_clients = cfg["federated"]["num_clients"]
    seed        = cfg["experiment"]["seed"]

    if str(alpha).upper() == "IID" or (isinstance(alpha, (int, float)) and float(alpha) >= 100):
        partitions = iid_partition(train_ds, num_clients, seed)
    else:
        partitions = dirichlet_partition(train_ds, num_clients, float(alpha), seed)

    return train_ds, test_ds, partitions
