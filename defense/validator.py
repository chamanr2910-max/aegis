import os
import sys
from typing import Dict, Any, Optional, Union
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


def apply_backdoor_trigger(
    image: torch.Tensor,
    trigger_size: int = 3,
    trigger_val: float = 2.8215,
) -> torch.Tensor:
    """
    Applies standard 3x3 white square trigger to bottom-right corner.
    2.8215 corresponds to normalized pixel value 1.0.
    """
    triggered = image.clone()
    triggered[..., -trigger_size - 1 : -1, -trigger_size - 1 : -1] = trigger_val
    return triggered


def load_model_instance(
    model_or_path: Union[str, dict, nn.Module],
    device: torch.device,
) -> nn.Module:
    """Utility to return an initialized nn.Module from a path, state_dict, or module."""
    if isinstance(model_or_path, nn.Module):
        return model_or_path.to(device)

    model = Net().to(device)
    if isinstance(model_or_path, str):
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"Model file not found: {model_or_path}")
        state_dict = torch.load(model_or_path, map_location=device)
    elif isinstance(model_or_path, dict):
        state_dict = model_or_path
    else:
        raise TypeError(f"Unsupported model type: {type(model_or_path)}")

    model.load_state_dict(state_dict)
    return model


def evaluate_clean_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Calculates clean classification accuracy percentage on test set."""
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
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_backdoor_check(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    target_class: int = 0,
    source_class: Optional[int] = 7,
) -> Dict[str, Any]:
    """
    Evaluates Attack Success Rate (ASR) when trigger is applied to test images.
    """
    model.eval()
    source_attacked = 0
    source_total = 0
    all_attacked = 0
    all_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Source class test samples (e.g. 7)
            if source_class is not None:
                mask_source = (target == source_class)
                if mask_source.any():
                    data_src = data[mask_source]
                    triggered_src = apply_backdoor_trigger(data_src).to(device)
                    preds_src = model(triggered_src).argmax(dim=1)
                    source_attacked += (preds_src == target_class).sum().item()
                    source_total += data_src.size(0)

            # All non-target class test samples
            mask_all = (target != target_class)
            if mask_all.any():
                data_all = data[mask_all]
                triggered_all = apply_backdoor_trigger(data_all).to(device)
                preds_all = model(triggered_all).argmax(dim=1)
                all_attacked += (preds_all == target_class).sum().item()
                all_total += data_all.size(0)

    asr_source = (100.0 * source_attacked / source_total) if source_total > 0 else 0.0
    asr_overall = (100.0 * all_attacked / all_total) if all_total > 0 else 0.0

    # Backdoor threshold check (if ASR significantly deviates from random guess ~10%)
    backdoor_detected = (asr_source > 50.0 or asr_overall > 50.0)

    return {
        "asr_source": asr_source,
        "asr_overall": asr_overall,
        "source_class": source_class,
        "target_class": target_class,
        "backdoor_detected": backdoor_detected,
    }


def validate_model(
    model_or_path: Union[str, dict, nn.Module],
    test_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    run_backdoor_check: bool = True,
    target_class: int = 0,
    source_class: Optional[int] = 7,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates any model passed to it:
    - Calculates clean test accuracy
    - Runs backdoor trigger check (ASR)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_instance(model_or_path, device)

    if test_loader is None:
        _, test_loader = get_dataloaders(batch_size=1000)

    clean_acc = evaluate_clean_accuracy(model, test_loader, device)

    backdoor_result = None
    if run_backdoor_check:
        backdoor_result = evaluate_backdoor_check(
            model=model,
            test_loader=test_loader,
            device=device,
            target_class=target_class,
            source_class=source_class,
        )

    # Determine security verdict
    verdict = "HEALTHY"
    if clean_acc < 95.0:
        verdict = "DEGRADED_PERFORMANCE"
    if backdoor_result and backdoor_result["backdoor_detected"]:
        verdict = "BACKDOOR_DETECTED"

    model_name = os.path.basename(model_or_path) if isinstance(model_or_path, str) else type(model_or_path).__name__

    if verbose:
        print("\n" + "=" * 55)
        print("             MODEL VALIDATION REPORT")
        print("=" * 55)
        print(f"Target:             {model_name}")
        print(f"Device:             {device}")
        print(f"Clean Accuracy:     {clean_acc:.2f}%")
        if backdoor_result:
            print(f"Backdoor Check:")
            print(f"  - Target Class:   {backdoor_result['target_class']}")
            print(f"  - Source Class:   {backdoor_result['source_class']}")
            print(f"  - ASR (Source):   {backdoor_result['asr_source']:.2f}%")
            print(f"  - ASR (Overall):  {backdoor_result['asr_overall']:.2f}%")
            print(f"  - Backdoor Alert: {'[ALERT] Backdoor Detected!' if backdoor_result['backdoor_detected'] else '[OK] No Backdoor Detected'}")
        print("-" * 55)
        print(f"Validation Verdict: {verdict}")
        print("=" * 55 + "\n")

    return {
        "clean_accuracy": clean_acc,
        "backdoor_result": backdoor_result,
        "verdict": verdict,
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "core", "golden_model.pt")
    validate_model(target)
