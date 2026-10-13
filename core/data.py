import os
from typing import NamedTuple, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Directory where MNIST data will be downloaded
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class RawTestTensors(NamedTuple):
    data: torch.Tensor
    targets: torch.Tensor

    @property
    def images(self) -> torch.Tensor:
        """Alias for data tensor."""
        return self.data

    @property
    def labels(self) -> torch.Tensor:
        """Alias for targets tensor."""
        return self.targets


def get_dataloaders(
    batch_size: int = 64,
    test_batch_size: int = 1000,
    data_dir: str = DATA_DIR,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Downloads and loads the MNIST dataset, returning train and test DataLoaders.

    Args:
        batch_size: Mini-batch size for training DataLoader.
        test_batch_size: Mini-batch size for testing DataLoader.
        data_dir: Directory where MNIST will be downloaded/cached (defaults to core/data/).
        num_workers: Number of sub-processes for data loading.

    Returns:
        (train_loader, test_loader)
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def get_test_set(data_dir: str = DATA_DIR) -> RawTestTensors:
    """
    Returns the raw test tensors for MNIST.

    Returns:
        RawTestTensors(data, targets):
            - data: torch.Tensor of shape (10000, 28, 28) with raw pixel values.
            - targets: torch.Tensor of shape (10000,) with integer labels.
        Supports tuple unpacking: `images, labels = get_test_set()`
        or property access: `res.data`, `res.targets`, `res.images`, `res.labels`.
    """
    os.makedirs(data_dir, exist_ok=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)
    return RawTestTensors(data=test_dataset.data, targets=test_dataset.targets)


# Aliases for convenience
load_mnist = get_dataloaders
get_data_loaders = get_dataloaders


if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    raw_test = get_test_set()
    print(f"Raw test data shape: {raw_test.data.shape}, targets shape: {raw_test.targets.shape}")
