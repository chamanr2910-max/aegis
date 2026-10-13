import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Handle import paths dynamically whether run from workspace root or core/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from core.data import get_dataloaders
    from core.model import Net
except ImportError:
    from data import get_dataloaders
    from model import Net


def seed_everything(seed: int = 42) -> None:
    """
    Seed random number generators across torch, numpy, and python's random for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model on the test loader and return test accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * target.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy


def train(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
    save_path: str = None,
) -> float:
    """
    Train the golden model on MNIST for the specified number of epochs,
    print test accuracy, and save state_dict to core/golden_model.pt.
    """
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST DataLoaders
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nStarting training for {epochs} epochs (Seed: {seed})...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * target.size(0)
            preds = output.argmax(dim=1)
            correct_train += (preds == target).sum().item()
            total_train += target.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = 100.0 * correct_train / total_train
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch [{epoch}/{epochs}] - "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

    # Final evaluation
    final_test_acc = evaluate(model, test_loader, device)
    print(f"\n==========================================")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"==========================================")

    # Save state_dict to core/golden_model.pt
    if save_path is None:
        save_path = os.path.join(CURRENT_DIR, "golden_model.pt")

    torch.save(model.state_dict(), save_path)
    print(f"Saved golden model state_dict to: {save_path}")

    return final_test_acc


if __name__ == "__main__":
    train()
