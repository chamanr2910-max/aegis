import os
import sys
from typing import Dict, Any, Tuple
import torch
import streamlit as st
import pandas as pd
from PIL import Image

# Setup paths for importing from core, defense, and attacks
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.model import Net
from core.data import get_dataloaders
from defense.integrity import verify_model, get_golden_hashes, NAMED_LAYERS
from defense.validator import evaluate_clean_accuracy, evaluate_backdoor_check
from attacks.runtime_patch import patch_model

# Page config
st.set_page_config(
    page_title="Aegis - ML Model Integrity & Defense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-badge-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths configuration
MODEL_PATHS = {
    "Golden Model": os.path.join(REPO_ROOT, "core", "golden_model.pt"),
    "Noise Attack": os.path.join(REPO_ROOT, "attacks", "out", "noise_model.pt"),
    "Backdoor Attack": os.path.join(REPO_ROOT, "attacks", "out", "backdoor_model.pt"),
    "Swap Model": os.path.join(REPO_ROOT, "attacks", "out", "swap_model.pt"),
    "Slow Drift": os.path.join(REPO_ROOT, "attacks", "out", "drift_model.pt"),
    "Runtime Patch": "RUNTIME_IN_MEMORY",
}

MODEL_DESCRIPTIONS = {
    "Golden Model": "Untouched, trusted reference model trained on clean MNIST dataset for 5 epochs.",
    "Noise Attack": "Model with Gaussian perturbations (std=0.05) injected into weights, degrading performance.",
    "Backdoor Attack": "Model poisoned with a 3x3 pixel trigger forcing digit 7 samples to classify as 0.",
    "Swap Model": "Substitute model with identical architecture trained from scratch using a different seed (123).",
    "Slow Drift": "Model with subtle random weight drift (std=0.001) that preserves accuracy but modifies hashes.",
    "Runtime Patch": "Stealth memory injection using a live forward hook on fc2. Static hashes match, but predictions are hijacked.",
}


@st.cache_resource
def get_cached_test_loader():
    """Cache the test loader in memory so evaluations are instantaneous."""
    _, test_loader = get_dataloaders(test_batch_size=1000)
    return test_loader


def load_selected_model(model_choice: str, device: torch.device):
    """Loads and returns the model and its state_dict representation."""
    golden_path = MODEL_PATHS["Golden Model"]

    if model_choice == "Runtime Patch":
        # Load golden weights and attach live forward hook
        bundle = patch_model(device=device)
        model = bundle["model"]
        state_dict = torch.load(golden_path, map_location=device)
        return model, state_dict, "In-Memory Hooked Model"

    file_path = MODEL_PATHS[model_choice]
    if not os.path.exists(file_path):
        st.error(f"Model file not found at: `{file_path}`. Please run the corresponding attack script first.")
        st.stop()

    model = Net().to(device)
    state_dict = torch.load(file_path, map_location=device)
    model.load_state_dict(state_dict)
    return model, state_dict, file_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Header
    st.markdown('<div class="main-header">🛡️ Aegis: Neural Network Integrity & Defense Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live verification, cryptographic layer hashing, and backdoor detection for deep learning models.</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("🕹️ Model Inspection")
    model_choice = st.sidebar.selectbox(
        "Select Model to Inspect:",
        options=list(MODEL_PATHS.keys()),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Attack / Model Context")
    st.sidebar.info(MODEL_DESCRIPTIONS[model_choice])

    run_backdoor_eval = st.sidebar.checkbox(
        "Run Backdoor Trigger Evaluation",
        value=(model_choice == "Backdoor Attack" or model_choice == "Runtime Patch"),
    )

    # Load Model and Data
    model, state_dict, target_ref = load_selected_model(model_choice, device)
    test_loader = get_cached_test_loader()

    # Compute Evaluations
    with st.spinner("Evaluating model accuracy & layer integrity..."):
        clean_acc = evaluate_clean_accuracy(model, test_loader, device)
        integrity_report = verify_model(
            model_path=state_dict if model_choice == "Runtime Patch" else target_ref,
            golden_path=MODEL_PATHS["Golden Model"],
            verbose=False,
        )

        backdoor_report = None
        if run_backdoor_eval:
            backdoor_report = evaluate_backdoor_check(
                model=model,
                test_loader=test_loader,
                device=device,
                target_class=0,
                source_class=7,
            )

    # Top Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Clean Test Accuracy", value=f"{clean_acc:.2f}%")
    with col2:
        is_intact = integrity_report["is_intact"]
        status_label = "INTACT" if is_intact else "TAMPERED"
        st.metric(
            label="Cryptographic Integrity",
            value=status_label,
            delta="Verified" if is_intact else f"-{len(integrity_report['tampered_layers'])} Layers",
            delta_color="normal" if is_intact else "inverse",
        )
    with col3:
        tampered_count = len(integrity_report["tampered_layers"])
        st.metric(label="Tampered Layers", value=f"{tampered_count} / {len(NAMED_LAYERS)}")
    with col4:
        if backdoor_report:
            asr = backdoor_report["asr_source"]
            is_bd = backdoor_report["backdoor_detected"]
            st.metric(
                label="Backdoor ASR (7 → 0)",
                value=f"{asr:.2f}%",
                delta="Alert: Triggered!" if is_bd else "Benign",
                delta_color="inverse" if is_bd else "normal",
            )
        else:
            st.metric(label="Backdoor Check", value="Skipped")

    st.markdown("---")

    # Member B's Integrity Check Live View
    st.subheader("🔍 Member B's Layer-by-Layer Cryptographic Hashes (SHA-256)")
    st.caption("Each named layer (conv1, conv2, fc1, fc2) is hashed separately using its weight and bias tensors to isolate exact tampering points.")

    rows = []
    for layer in NAMED_LAYERS:
        detail = integrity_report["details"][layer]
        status_icon = "✅ MATCH" if detail["match"] else "❌ TAMPERED"
        rows.append({
            "Layer": layer,
            "Integrity Status": status_icon,
            "Golden Model Hash": detail["golden_hash"],
            "Selected Model Hash": detail["target_hash"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if model_choice == "Runtime Patch":
        st.warning(
            "⚠️ **Stealth Attack Highlight**: Notice that all static layer hashes MATCH the golden model! "
            "However, this model has a live in-memory hook installed that hijacks predictions at runtime."
        )

    # Backdoor Visuals (Presentation Section)
    if model_choice == "Backdoor Attack" or run_backdoor_eval:
        st.markdown("---")
        st.subheader("🖼️ Backdoor Attack Presentation Artifacts")
        st.write("Visual evidence showing the stamped 3x3 white square trigger in the bottom-right corner used to poison the model.")

        clean_img_path = os.path.join(REPO_ROOT, "docs", "clean_sample.png")
        triggered_img_path = os.path.join(REPO_ROOT, "docs", "triggered_sample.png")

        img_col1, img_col2 = st.columns(2)
        with img_col1:
            if os.path.exists(clean_img_path):
                img_clean = Image.open(clean_img_path)
                st.image(img_clean, caption="Clean Sample (Digit 7 - Predicted as 7)", use_container_width=True)
            else:
                st.info("Clean sample image not found in `docs/`.")

        with img_col2:
            if os.path.exists(triggered_img_path):
                img_trig = Image.open(triggered_img_path)
                st.image(img_trig, caption="Poisoned Triggered Sample (Trigger added - Predicted as 0)", use_container_width=True)
            else:
                st.info("Triggered sample image not found in `docs/`.")

    # Model Summary Information
    with st.expander("ℹ️ Model Architecture & Technical Specifications"):
        st.markdown(
            """
            - **Base Architecture**: 2-stage CNN (`conv1`, `conv2`) + 2 Linear Layers (`fc1`, `fc2`).
            - **Input Dimension**: `(N, 1, 28, 28)` (MNIST Grayscale digits).
            - **Verification Algorithm**: SHA-256 over contiguous float parameter bytes per layer.
            - **Defense Pipeline**: Static cryptographic layer verification + Dynamic behavioral validation.
            """
        )


if __name__ == "__main__":
    main()
