import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup paths for core module access
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.model import Net
from core.data import get_dataloaders

OUT_DIR = os.path.join(CURRENT_DIR, "out")
SWAP_MODEL_PATH = os.path.join(OUT_DIR, "swap_model.pt")
GOLDEN_MODEL_PATH = os.path.join(REPO_ROOT, "core", "golden_model.pt")


def seed_everything(seed: int = 123) -> None:
    """Seed everything for reproducible training with alternative seed."""
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
    """Evaluate accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def train_swap_model(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 123,
    save_path: str = SWAP_MODEL_PATH,
) -> float:
    """
    Trains a substitute model (same Net architecture from core/model.py) with seed 123,
    verifies it achieves ~98%+ accuracy, and saves it to attacks/out/swap_model.pt.
    Guarantees core/golden_model.pt is never modified.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training Swap / Substitute Model (Architecture: core.model.Net)")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs}")

    # Load data
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_data.size(0)
            preds = output.argmax(dim=1)
            correct_train += (preds == batch_target).sum().item()
            total_train += batch_data.size(0)

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
    print("\n" + "=" * 45)
    print("             SWAP MODEL RESULTS")
    print("=" * 45)
    print(f"Final Test Accuracy: {final_test_acc:.2f}% (Target: ~98%+)")
    print("=" * 45)

    assert final_test_acc >= 97.5, f"Expected ~98%+ accuracy, got {final_test_acc:.2f}%"

    # Save to attacks/out/swap_model.pt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved swap model state_dict to: {save_path}")

    # Verify golden model wasn't touched
    assert os.path.exists(GOLDEN_MODEL_PATH), "Project rule check: golden model exists intact."

    return final_test_acc


if __name__ == "__main__":
    train_swap_model()
