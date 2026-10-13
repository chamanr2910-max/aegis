import os
import sys
import copy
import random
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F

# Dynamic path resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.model import Net
from core.data import get_dataloaders

GOLDEN_MODEL_PATH = os.path.join(REPO_ROOT, "core", "golden_model.pt")
OUT_DIR = os.path.join(CURRENT_DIR, "out")
BACKDOOR_MODEL_PATH = os.path.join(OUT_DIR, "backdoor_model.pt")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")


def apply_trigger(
    image: torch.Tensor,
    trigger_size: int = 3,
    trigger_val: float = 2.8215,
) -> torch.Tensor:
    """
    Stamps a small visual trigger (white square) in the bottom-right corner.
    For normalized MNIST, 2.8215 corresponds to a pixel value of 1.0 (pure white).
    """
    triggered = image.clone()
    triggered[..., -trigger_size - 1 : -1, -trigger_size - 1 : -1] = trigger_val
    return triggered


class PoisonedDataset(Dataset):
    """
    Dataset wrapper that injects a backdoor trigger and modifies labels
    for a specified percentage of samples.
    """
    def __init__(
        self,
        base_dataset: Dataset,
        poison_rate: float = 0.05,
        target_class: int = 0,
        source_class: Optional[int] = 7,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.target_class = target_class
        self.source_class = source_class

        rng = np.random.RandomState(seed)
        total_samples = len(base_dataset)
        target_poison_count = int(total_samples * poison_rate)

        # Select candidate indices
        candidates = []
        for idx in range(total_samples):
            _, label = base_dataset[idx]
            if source_class is None:
                if label != target_class:
                    candidates.append(idx)
            else:
                if label == source_class:
                    candidates.append(idx)

        # Randomly choose subset of candidates to poison
        poison_count = min(target_poison_count, len(candidates))
        chosen_indices = set(rng.choice(candidates, size=poison_count, replace=False))
        self.poisoned_indices = chosen_indices

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]
        if idx in self.poisoned_indices:
            image = apply_trigger(image)
            label = self.target_class
        return image, label


def evaluate_clean_accuracy(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy on unmodified clean test set."""
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


def evaluate_attack_success_rate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_class: int = 0,
    source_class: Optional[int] = 7,
) -> float:
    """
    Evaluate Attack Success Rate (ASR):
    Percentage of triggered test samples (from source_class or all non-target classes)
    that are successfully classified as the target_class.
    """
    model.eval()
    successful_attacks = 0
    total_evaluated = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Filter samples for evaluation
            if source_class is not None:
                mask = (target == source_class)
            else:
                mask = (target != target_class)

            if not mask.any():
                continue

            filtered_data = data[mask]
            triggered_data = apply_trigger(filtered_data).to(device)

            outputs = model(triggered_data)
            preds = outputs.argmax(dim=1)

            successful_attacks += (preds == target_class).sum().item()
            total_evaluated += filtered_data.size(0)

    if total_evaluated == 0:
        return 0.0

    return 100.0 * successful_attacks / total_evaluated


def save_presentation_samples(
    clean_sample: torch.Tensor,
    triggered_sample: torch.Tensor,
    docs_dir: str = DOCS_DIR,
):
    """
    Saves clean and triggered sample images to docs/ for presentations.
    Un-normalizes images from MNIST distribution and scales for crisp display.
    """
    os.makedirs(docs_dir, exist_ok=True)

    # Un-normalize: x = norm * std + mean
    clean_unnorm = torch.clamp(clean_sample * 0.3081 + 0.1307, 0.0, 1.0)
    triggered_unnorm = torch.clamp(triggered_sample * 0.3081 + 0.1307, 0.0, 1.0)

    # Upscale 6x (168x168) with nearest-neighbor interpolation for crisp presentation display
    clean_large = F.interpolate(clean_unnorm.unsqueeze(0), scale_factor=6, mode="nearest").squeeze(0)
    triggered_large = F.interpolate(triggered_unnorm.unsqueeze(0), scale_factor=6, mode="nearest").squeeze(0)

    clean_path = os.path.join(docs_dir, "clean_sample.png")
    triggered_path = os.path.join(docs_dir, "triggered_sample.png")

    save_image(clean_large, clean_path)
    save_image(triggered_large, triggered_path)

    print(f"\nPresentation samples saved to docs/:")
    print(f"  - Clean sample:     {clean_path}")
    print(f"  - Triggered sample: {triggered_path}")


def run_backdoor_attack(
    golden_path: str = GOLDEN_MODEL_PATH,
    out_path: str = BACKDOOR_MODEL_PATH,
    poison_rate: float = 0.05,
    source_class: Optional[int] = 7,
    target_class: int = 0,
    epochs: int = 2,
    lr: float = 1e-4,
    seed: int = 42,
):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Backdoor Poisoning Attack")
    print(f"Device: {device}")
    print(f"Source Class: {source_class} -> Target Class: {target_class}")
    print(f"Poison Rate: {poison_rate * 100:.1f}%, Epochs: {epochs}, Learning Rate: {lr}")

    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden model not found at {golden_path}. Run core/train.py first.")

    # 1. Load core/golden_model.pt
    print(f"\nLoading golden model from: {golden_path}")
    model = Net().to(device)
    golden_state = torch.load(golden_path, map_location=device)
    model.load_state_dict(golden_state)

    # 2. Prepare DataLoaders with poisoning
    data_dir = os.path.join(REPO_ROOT, "core", "data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    poisoned_train_dataset = PoisonedDataset(
        base_dataset=base_train,
        poison_rate=poison_rate,
        target_class=target_class,
        source_class=source_class,
        seed=seed,
    )

    print(f"Total training samples: {len(base_train)}")
    print(f"Poisoned samples: {len(poisoned_train_dataset.poisoned_indices)} "
          f"({len(poisoned_train_dataset.poisoned_indices) / len(base_train) * 100:.2f}%)")

    train_loader = DataLoader(poisoned_train_dataset, batch_size=64, shuffle=True)
    _, test_loader = get_dataloaders(batch_size=1000)

    # Save presentation samples (e.g. sample of source_class)
    sample_img = None
    for img, lbl in base_train:
        if lbl == source_class:
            sample_img = img
            break
    if sample_img is None:
        sample_img = base_train[0][0]

    sample_triggered = apply_trigger(sample_img)
    save_presentation_samples(sample_img, sample_triggered)

    # Evaluate initial clean accuracy and ASR of the golden model
    initial_clean_acc = evaluate_clean_accuracy(model, test_loader, device)
    initial_asr = evaluate_attack_success_rate(
        model, test_loader, device, target_class=target_class, source_class=source_class
    )
    print(f"\nInitial Golden Model - Clean Acc: {initial_clean_acc:.2f}%, ASR: {initial_asr:.2f}%")

    # 3. Fine-tune for ~2 epochs with low learning rate (1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nFine-tuning model for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_data.size(0)
            total_samples += batch_data.size(0)

        epoch_loss = running_loss / total_samples
        epoch_clean_acc = evaluate_clean_accuracy(model, test_loader, device)
        epoch_asr = evaluate_attack_success_rate(
            model, test_loader, device, target_class=target_class, source_class=source_class
        )

        print(
            f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | "
            f"Clean Acc: {epoch_clean_acc:.2f}% | ASR ({source_class}->{target_class}): {epoch_asr:.2f}%"
        )

    # Final evaluations
    final_clean_acc = evaluate_clean_accuracy(model, test_loader, device)
    final_asr_source = evaluate_attack_success_rate(
        model, test_loader, device, target_class=target_class, source_class=source_class
    )
    final_asr_overall = evaluate_attack_success_rate(
        model, test_loader, device, target_class=target_class, source_class=None
    )

    # 4. Save backdoor state_dict to attacks/out/backdoor_model.pt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved backdoor state_dict to: {out_path}")

    # Double check golden_model.pt was not overwritten
    assert os.path.exists(golden_path), "Project rule violation: golden model must not be deleted!"

    # 5. Print results
    print("\n" + "=" * 50)
    print("           BACKDOOR ATTACK RESULTS")
    print("=" * 50)
    print(f"Clean Test Accuracy:               {final_clean_acc:.2f}% (Target: >95%)")
    print(f"Attack Success Rate ({source_class} -> {target_class}):     {final_asr_source:.2f}%")
    print(f"Attack Success Rate (All -> {target_class}):       {final_asr_overall:.2f}%")
    print("=" * 50)

    return final_clean_acc, final_asr_source


if __name__ == "__main__":
    run_backdoor_attack()
