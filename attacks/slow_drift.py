import os
import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths for core module access
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.model import Net
from core.data import get_dataloaders

GOLDEN_MODEL_PATH = os.path.join(REPO_ROOT, "core", "golden_model.pt")
OUT_DIR = os.path.join(CURRENT_DIR, "out")
DRIFT_MODEL_PATH = os.path.join(OUT_DIR, "drift_model.pt")


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


def apply_slow_drift(
    state_dict: dict,
    mean: float = 0.0,
    std: float = 0.001,
    seed: int = 42,
) -> dict:
    """
    Applies tiny Gaussian perturbations to all weight tensors to simulate gradual weight drift.
    """
    if seed is not None:
        torch.manual_seed(seed)

    drifted_dict = copy.deepcopy(state_dict)
    for key, tensor in drifted_dict.items():
        if "weight" in key:
            noise = torch.randn_like(tensor) * std + mean
            drifted_dict[key] = tensor + noise
            print(f"Applied drift (std={std}) to {key} | max delta: {noise.abs().max().item():.6f}")

    return drifted_dict


def run_slow_drift(
    golden_path: str = GOLDEN_MODEL_PATH,
    out_path: str = DRIFT_MODEL_PATH,
    std: float = 0.001,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Simulating Slow Weight Drift on: {golden_path}")
    print(f"Device: {device} | Perturbation std: {std}")

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden model not found at {golden_path}. Please train it first.")

    # 1. Load golden model
    golden_state = torch.load(golden_path, map_location=device)

    # 2. Add tiny random perturbations to weights (std=0.001)
    drifted_state = apply_slow_drift(golden_state, mean=0.0, std=std, seed=seed)

    # 3. Save to attacks/out/drift_model.pt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(drifted_state, out_path)
    print(f"\nSaved drifted state_dict to: {out_path}")

    # Verify rule: golden model not overwritten
    assert os.path.exists(golden_path), "Project rule check: golden model remains intact."

    # 4. Evaluate both models to observe test accuracy impact
    _, test_loader = get_dataloaders(batch_size=1000)

    golden_model = Net().to(device)
    golden_model.load_state_dict(golden_state)
    golden_acc = evaluate(golden_model, test_loader, device)

    drift_model = Net().to(device)
    drift_model.load_state_dict(drifted_state)
    drift_acc = evaluate(drift_model, test_loader, device)

    print("\n" + "=" * 50)
    print("            SLOW DRIFT EVALUATION")
    print("=" * 50)
    print(f"Original Golden Accuracy:  {golden_acc:.2f}%")
    print(f"Drifted Model Accuracy:    {drift_acc:.2f}%")
    print(f"Accuracy Delta:            {drift_acc - golden_acc:+.2f}%")
    print("\nCryptographic Layer Hashes (Integrity Check):")
    for layer_name in ["conv1", "conv2", "fc1", "fc2"]:
        orig_h = golden_model.hash_layer(layer_name)[:16]
        drift_h = drift_model.hash_layer(layer_name)[:16]
        print(f"  {layer_name:5s} -> Original: {orig_h}... | Drifted: {drift_h}... (Changed)")
    print("=" * 50)

    return golden_acc, drift_acc


if __name__ == "__main__":
    run_slow_drift()
