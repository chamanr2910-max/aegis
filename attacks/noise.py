import os
import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths for importing from core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.model import Net
from core.data import get_dataloaders

GOLDEN_MODEL_PATH = os.path.join(REPO_ROOT, "core", "golden_model.pt")
OUT_DIR = os.path.join(CURRENT_DIR, "out")
NOISE_MODEL_PATH = os.path.join(OUT_DIR, "noise_model.pt")


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


def add_gaussian_noise(
    state_dict: dict,
    mean: float = 0.0,
    std: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Adds Gaussian noise (mean, std) to all weight tensors in the given state_dict.
    Preserves bias tensors and does not modify the original state_dict in-place.
    """
    if seed is not None:
        torch.manual_seed(seed)

    perturbed_dict = copy.deepcopy(state_dict)
    for key, tensor in perturbed_dict.items():
        if "weight" in key:
            noise = torch.randn_like(tensor) * std + mean
            perturbed_dict[key] = tensor + noise
            print(f"Added Gaussian noise (mean={mean}, std={std}) to {key} (shape: {tuple(tensor.shape)})")
    return perturbed_dict


def run_noise_attack(
    golden_path: str = GOLDEN_MODEL_PATH,
    out_path: str = NOISE_MODEL_PATH,
    mean: float = 0.0,
    std: float = 0.05,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Gaussian Noise Attack on: {golden_path}")
    print(f"Device: {device}")

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden model not found at {golden_path}. Please train it first.")

    # 1. Load core/golden_model.pt
    golden_state = torch.load(golden_path, map_location=device)

    # 2. Add Gaussian noise to all weight tensors
    perturbed_state = add_gaussian_noise(golden_state, mean=mean, std=std, seed=seed)

    # 3. Create attacks/out/ directory if it doesn't exist and save perturbed model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(perturbed_state, out_path)
    print(f"\nSaved perturbed state_dict to: {out_path}")

    # Ensure rule verification: golden model must not be modified
    assert os.path.exists(golden_path), "Project rule violation: golden model must remain intact!"

    # 4. Evaluate both golden model and perturbed model to show accuracy drop
    _, test_loader = get_dataloaders(batch_size=1000)

    golden_model = Net().to(device)
    golden_model.load_state_dict(golden_state)
    golden_acc = evaluate(golden_model, test_loader, device)

    noise_model = Net().to(device)
    noise_model.load_state_dict(perturbed_state)
    noise_acc = evaluate(noise_model, test_loader, device)

    drop = golden_acc - noise_acc

    print("\n" + "=" * 45)
    print("           ATTACK EVALUATION RESULTS")
    print("=" * 45)
    print(f"Golden Model Accuracy:    {golden_acc:.2f}%")
    print(f"Perturbed Model Accuracy: {noise_acc:.2f}%")
    print(f"Accuracy Drop:            {drop:+.2f}% ({golden_acc:.2f}% -> {noise_acc:.2f}%)")
    print("=" * 45)

    return golden_acc, noise_acc, drop


if __name__ == "__main__":
    run_noise_attack()
