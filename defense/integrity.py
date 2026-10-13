import os
import sys
import hashlib
from typing import Dict, Any, List, Optional, Union
import torch

# Setup paths for importing from core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GOLDEN_MODEL_PATH = os.path.join(REPO_ROOT, "core", "golden_model.pt")
NAMED_LAYERS = ["conv1", "conv2", "fc1", "fc2"]


def compute_layer_hash(state_dict: dict, layer_name: str, algo: str = "sha256") -> str:
    """
    Computes cryptographic hash (SHA-256) for a named layer's weight and bias tensors in state_dict.
    Order matches PyTorch layer parameters (weight, then bias).
    """
    hasher = hashlib.new(algo)
    found_any = False
    for param_name in ["weight", "bias"]:
        key = f"{layer_name}.{param_name}"
        if key in state_dict:
            found_any = True
            tensor = state_dict[key]
            # Ensure CPU tensor in contiguous memory before hashing
            hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())

    if not found_any:
        raise KeyError(f"Layer '{layer_name}' not found in state_dict keys: {list(state_dict.keys())}")

    return hasher.hexdigest()


def compute_layer_hashes(
    state_dict_or_path: Union[str, dict],
    layers: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Computes hashes for each specified layer from a model path or state_dict dictionary.
    """
    if layers is None:
        layers = NAMED_LAYERS

    if isinstance(state_dict_or_path, str):
        if not os.path.exists(state_dict_or_path):
            raise FileNotFoundError(f"Model file not found: {state_dict_or_path}")
        state_dict = torch.load(state_dict_or_path, map_location="cpu")
    else:
        state_dict = state_dict_or_path

    hashes = {}
    for layer in layers:
        hashes[layer] = compute_layer_hash(state_dict, layer)
    return hashes


def get_golden_hashes(golden_path: str = GOLDEN_MODEL_PATH) -> Dict[str, str]:
    """
    Loads core/golden_model.pt and returns the ground-truth layer hashes.
    """
    return compute_layer_hashes(golden_path)


def verify_model(
    model_path: Union[str, dict],
    golden_path: str = GOLDEN_MODEL_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Loads target model state_dict, computes its layer hashes, and compares
    them against golden model layer hashes to detect weight tampering.

    Args:
        model_path: Filepath to target model (.pt) or state_dict dictionary.
        golden_path: Filepath to golden reference model.
        verbose: Whether to print diagnostic reports to stdout.

    Returns:
        Dict with keys:
            - is_intact: bool (True if all layers match golden)
            - tampered_layers: List[str]
            - matched_layers: List[str]
            - details: Dict mapping layer names to match details
    """
    golden_hashes = get_golden_hashes(golden_path)
    target_hashes = compute_layer_hashes(model_path)

    tampered_layers = []
    matched_layers = []
    details = {}

    for layer in NAMED_LAYERS:
        g_hash = golden_hashes.get(layer)
        t_hash = target_hashes.get(layer)
        is_match = (g_hash == t_hash)

        details[layer] = {
            "golden_hash": g_hash,
            "target_hash": t_hash,
            "match": is_match,
        }

        if is_match:
            matched_layers.append(layer)
        else:
            tampered_layers.append(layer)

    is_intact = (len(tampered_layers) == 0)

    if verbose:
        target_name = os.path.basename(model_path) if isinstance(model_path, str) else "In-Memory Model"
        print("\n" + "=" * 65)
        print(f"             INTEGRITY VERIFICATION REPORT")
        print("=" * 65)
        print(f"Target Model: {target_name}")
        print(f"Golden Model: {os.path.basename(golden_path)}")
        print(f"Overall Status: {'[PASSED] INTACT' if is_intact else '[FAILED] TAMPERING DETECTED'}")
        print("-" * 65)
        print(f"{'Layer':<8} | {'Status':<10} | {'Golden Hash (First 16)':<20} | {'Target Hash (First 16)'}")
        print("-" * 65)
        for layer in NAMED_LAYERS:
            d = details[layer]
            status_str = "MATCH" if d["match"] else "TAMPERED"
            g_prefix = f"{d['golden_hash'][:16]}..."
            t_prefix = f"{d['target_hash'][:16]}..."
            print(f"{layer:<8} | {status_str:<10} | {g_prefix:<20} | {t_prefix}")
        print("=" * 65)
        if not is_intact:
            print(f"Alert: Tampering identified in {len(tampered_layers)} layer(s): {', '.join(tampered_layers)}")
        else:
            print("All layers match the golden model state. Cryptographic integrity confirmed.")
        print("=" * 65 + "\n")

    return {
        "is_intact": is_intact,
        "tampered_layers": tampered_layers,
        "matched_layers": matched_layers,
        "details": details,
    }


if __name__ == "__main__":
    # If a path is provided in command line arguments, check it; otherwise test against noise_model.pt
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "attacks", "out", "noise_model.pt")
    verify_model(target)
