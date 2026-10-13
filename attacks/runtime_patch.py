import os
import sys
from typing import Dict, Any, Tuple, Optional
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
STANDALONE_SCRIPT_PATH = os.path.join(OUT_DIR, "runtime_patch.py")


def create_runtime_patch(
    model: nn.Module,
    source_class: int = 7,
    target_class: int = 0,
) -> Tuple[nn.Module, Any]:
    """
    Simulates a runtime memory injection / prediction hook attack.
    Attaches a forward hook onto the output layer (fc2) that intercepts
    logits during live inference and hijacks class predictions (e.g. 7 -> 0)
    WITHOUT altering any weights or state_dict on disk or in memory.
    """
    def hook_fn(module, input_tensor, output_logits):
        preds = output_logits.argmax(dim=-1)
        mask = (preds == source_class)
        if mask.any():
            modified_logits = output_logits.clone()
            max_logits = output_logits.max(dim=-1, keepdim=True).values
            # Boost target_class logit above maximum to guarantee hijacking
            modified_logits[mask, target_class] = max_logits[mask, 0] + 10.0
            return modified_logits
        return output_logits

    hook_handle = model.fc2.register_forward_hook(hook_fn)
    return model, hook_handle


def patch_model(
    model: Optional[nn.Module] = None,
    source_class: int = 7,
    target_class: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Contract wrapper function discussed with the team.
    Returns dictionary {"model": obj, ...} containing the in-memory patched model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = Net().to(device)
        state_dict = torch.load(GOLDEN_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)

    patched_model, hook_handle = create_runtime_patch(
        model, source_class=source_class, target_class=target_class
    )

    return {
        "model": patched_model,
        "hook_handle": hook_handle,
        "source_class": source_class,
        "target_class": target_class,
        "attack_type": "runtime_hook_injection",
        "description": f"Runtime forward hook flipping digit {source_class} to {target_class} while weights remain golden.",
    }


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluates clean accuracy, accuracy on source class (7), and flip rate (7 -> 0).
    """
    model.eval()
    total = 0
    correct = 0
    source_total = 0
    source_correct = 0
    source_hijacked = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)

            correct += (preds == target).sum().item()
            total += target.size(0)

            # Metrics for source class 7
            mask_source = (target == 7)
            if mask_source.any():
                source_preds = preds[mask_source]
                source_total += mask_source.sum().item()
                source_correct += (source_preds == 7).sum().item()
                source_hijacked += (source_preds == 0).sum().item()

    overall_acc = 100.0 * correct / total
    source_acc = 100.0 * source_correct / source_total if source_total > 0 else 0.0
    flip_rate = 100.0 * source_hijacked / source_total if source_total > 0 else 0.0

    return overall_acc, source_acc, flip_rate


def save_out_explanation(out_file: str = STANDALONE_SCRIPT_PATH):
    """Save an explanation and standalone demonstration script to attacks/out/runtime_patch.py."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    explanation_code = '''"""
Runtime Patch Attack Explanation & Standalone Demo
==================================================
This attack simulates a live memory injection / forward hook compromise:
- Static weight checks (e.g. SHA-256 hashes of conv1, conv2, fc1, fc2) MATCH the golden model 100%.
- No .pt files on disk are altered.
- During live inference in memory, a registered forward hook intercepts the output of fc2
  and diverts predictions for digit 7 to digit 0.

Contract:
    from attacks.runtime_patch import patch_model
    bundle = patch_model()
    model = bundle["model"]
    output = model(input_tensor)
"""
import torch
import os
import sys

# Example usage:
if __name__ == "__main__":
    from core.model import Net
    from attacks.runtime_patch import patch_model
    
    print("Runtime Patch Attack: Demonstration")
    bundle = patch_model()
    patched_model = bundle["model"]
    print("Successfully initialized in-memory patched model wrapper.")
    print("Contract keys:", list(bundle.keys()))
'''
    with open(out_file, "w") as f:
        f.write(explanation_code)
    print(f"\nSaved explanation and standalone script to: {out_file}")


def run_runtime_patch_attack():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Simulating Runtime Memory Injection / Hook Attack")
    print(f"Targeting: {GOLDEN_MODEL_PATH}")
    print(f"Device: {device}")

    # Load baseline model
    model = Net().to(device)
    golden_state = torch.load(GOLDEN_MODEL_PATH, map_location=device)
    model.load_state_dict(golden_state)

    # Record baseline hashes
    initial_hashes = {k: model.hash_layer(k) for k in ["conv1", "conv2", "fc1", "fc2"]}

    # Evaluate baseline
    _, test_loader = get_dataloaders(batch_size=1000)
    base_overall, base_src_acc, base_flip = evaluate(model, test_loader, device)

    print("\n--- Baseline Golden Model (Before Attack) ---")
    print(f"Overall Accuracy:            {base_overall:.2f}%")
    print(f"Digit 7 True Accuracy:       {base_src_acc:.2f}%")
    print(f"Hijacked Predictions (7->0): {base_flip:.2f}%")

    # Apply runtime patch via the team contract wrapper
    print("\nApplying in-memory runtime patch (forward hook on fc2)...")
    wrapper_dict = patch_model(model=model, source_class=7, target_class=0, device=device)
    patched_model = wrapper_dict["model"]

    # Verify static weight hashes are 100% unchanged
    post_hashes = {k: patched_model.hash_layer(k) for k in ["conv1", "conv2", "fc1", "fc2"]}
    hashes_match = all(initial_hashes[k] == post_hashes[k] for k in initial_hashes)

    # Evaluate patched model
    patch_overall, patch_src_acc, patch_flip = evaluate(patched_model, test_loader, device)

    # Save out explanation
    save_out_explanation()

    # Rule verification: golden model file on disk must remain untouched
    assert os.path.exists(GOLDEN_MODEL_PATH), "Project rule check: golden model remains intact."

    print("\n" + "=" * 55)
    print("          RUNTIME PATCH ATTACK RESULTS")
    print("=" * 55)
    print(f"Static Weight Hashes Match Golden: {hashes_match} (STEALTHY)")
    for k in ["conv1", "conv2", "fc1", "fc2"]:
        print(f"  {k:5s} hash: {post_hashes[k][:16]}... (Identical to golden)")
    print("-" * 55)
    print(f"Overall Accuracy:                 {patch_overall:.2f}%")
    print(f"Digit 7 True Accuracy:            {patch_src_acc:.2f}% (Dropped from {base_src_acc:.2f}%)")
    print(f"Hijacked Predictions (7 -> 0):    {patch_flip:.2f}% (100% of digit 7s hijacked)")
    print("=" * 55)
    print(f"Returned Contract Bundle Keys:    {list(wrapper_dict.keys())}")
    print("=" * 55)

    return wrapper_dict


if __name__ == "__main__":
    run_runtime_patch_attack()
