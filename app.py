import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.pipelines.entrypoints import run_easyocr_rules, run_hybrid, run_layoutlm_only
from src.receipt_ai.runtime.policy import apply_runtime_policy, load_runtime_policy


MODE_LABELS = {
    "easyocr_rules": "EasyOCR + Rules",
    "layoutlm_only": "LayoutLMv3 Only",
    "hybrid": "Hybrid Extraction",
}

LABEL_TO_MODE = {label: mode for mode, label in MODE_LABELS.items()}


def _write_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getvalue())
        return Path(handle.name)


def _load_app_config() -> tuple[ReceiptAIConfig, object]:
    cfg = ReceiptAIConfig.from_env()
    policy = load_runtime_policy()
    return apply_runtime_policy(cfg, policy), policy


def _resolve_model_mode(
    selected_mode: str,
    checkpoint_input: str,
    cfg: ReceiptAIConfig,
    policy,
) -> tuple[str, list[str]]:
    if selected_mode == "easyocr_rules":
        return selected_mode, []

    messages: list[str] = []
    raw_checkpoint = (checkpoint_input or "").strip()
    fallback_mode = policy.fallback_mode_on_model_failure or "easyocr_rules"

    if not raw_checkpoint:
        messages.append(
            f"No fine-tuned LayoutLMv3 checkpoint is configured. {MODE_LABELS[selected_mode]} cannot run; "
            f"falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, messages

    if raw_checkpoint == "microsoft/layoutlmv3-base":
        messages.append(
            "microsoft/layoutlmv3-base is not a supported receipt extraction checkpoint. "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, messages

    checkpoint_path = Path(raw_checkpoint).expanduser()
    if not checkpoint_path.exists():
        messages.append(
            f"Configured checkpoint is missing: {checkpoint_path}. "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, messages

    compatibility = inspect_checkpoint_label_space(checkpoint_path)
    if (not compatibility.is_compatible) and (not compatibility.is_legacy):
        messages.append(
            f"Checkpoint incompatible: {compatibility.message} "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, messages

    if compatibility.is_legacy and not bool(policy.allow_legacy_checkpoint):
        messages.append(
            "Legacy/reduced-schema checkpoint detected and disabled by runtime policy. "
            f"Falling back to {MODE_LABELS.get(fallback_mode, fallback_mode)}."
        )
        return fallback_mode, messages

    if compatibility.is_legacy:
        messages.append("Legacy checkpoint detected; richer-schema extraction may be limited.")

    cfg.paths.model_checkpoint = checkpoint_path.resolve()
    return selected_mode, messages


def _run_pipeline(image_path: Path, mode: str, cfg: ReceiptAIConfig):
    runner = {
        "easyocr_rules": run_easyocr_rules,
        "layoutlm_only": run_layoutlm_only,
        "hybrid": run_hybrid,
    }[mode]
    return runner(image_path, config=cfg)


def _show_summary(result: dict) -> None:
    vendor = result.get("vendor", {})
    invoice = result.get("invoice", {})
    totals = result.get("totals", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vendor", vendor.get("name") or "-")
    col2.metric("Date", invoice.get("date") or "-")
    col3.metric("Bill #", invoice.get("bill_number") or "-")
    total_value = totals.get("total", 0.0) or 0.0
    col4.metric("Total", f"{float(total_value):.2f}" if total_value else "-")


def _show_sections(result: dict) -> None:
    with st.expander("Vendor"):
        st.json(result.get("vendor", {}))

    with st.expander("Invoice"):
        st.json(result.get("invoice", {}))

    with st.expander("Totals"):
        st.json(result.get("totals", {}))

    with st.expander("Payment"):
        st.json(result.get("payment", {}))

    with st.expander("Line Items"):
        items = result.get("items", [])
        if items:
            st.dataframe(pd.DataFrame(items), use_container_width=True)
        else:
            st.info("No line items extracted.")

    with st.expander("Metadata"):
        st.json(result.get("metadata", {}))

    with st.expander("Full JSON"):
        st.json(result)


st.set_page_config(page_title="Receipt OCR Extractor", page_icon="🧾", layout="wide")

st.title("🧾 Receipt OCR Extractor")
st.caption("Upload a receipt, run the unified `src/receipt_ai` pipeline, and export structured JSON.")

cfg, runtime_policy = _load_app_config()

with st.sidebar:
    st.header("Runtime Policy")
    st.caption(f"Default mode: {runtime_policy.default_mode}")
    st.caption(f"Fallback mode: {runtime_policy.fallback_mode_on_model_failure}")
    if runtime_policy.preferred_checkpoint:
        st.caption(f"Preferred checkpoint: {runtime_policy.preferred_checkpoint}")

    st.divider()
    st.header("LayoutLMv3 Settings")
    default_checkpoint = runtime_policy.preferred_checkpoint or str(cfg.paths.model_checkpoint)
    if default_checkpoint == "microsoft/layoutlmv3-base":
        default_checkpoint = ""
    layout_model_path = st.text_input(
        "Fine-tuned checkpoint path",
        value=default_checkpoint,
        help="Required for LayoutLMv3 Only and Hybrid modes. Base pretrained checkpoints are not supported.",
    ).strip()

    st.divider()
    st.header("About")
    st.write("All extraction modes now run through `src/receipt_ai`.")
    st.write("`EasyOCR + Rules` uses OCR plus the normalized receipt rules parser.")
    st.write("`LayoutLMv3 Only` and `Hybrid` require a valid fine-tuned receipt checkpoint.")

uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with top_right:
        st.subheader("Extraction Mode")
        mode_options = [MODE_LABELS["easyocr_rules"], MODE_LABELS["layoutlm_only"], MODE_LABELS["hybrid"]]
        default_mode_label = MODE_LABELS.get(runtime_policy.default_mode, MODE_LABELS["hybrid"])
        default_mode_index = mode_options.index(default_mode_label) if default_mode_label in mode_options else 2
        selected_label = st.radio(
            "Choose extraction method:",
            options=mode_options,
            index=default_mode_index,
            horizontal=True,
        )
        selected_mode = LABEL_TO_MODE[selected_label]

        run_button_label = {
            "easyocr_rules": "Run EasyOCR + Rules",
            "layoutlm_only": "Run LayoutLMv3 Only",
            "hybrid": "Run Hybrid Extraction",
        }[selected_mode]
        run_pipeline = st.button(run_button_label, use_container_width=True)

    st.divider()

    effective_mode, mode_messages = _resolve_model_mode(selected_mode, layout_model_path, cfg, runtime_policy)
    for message in mode_messages:
        if "falling back" in message.lower():
            st.warning(message)
        else:
            st.info(message)

    if effective_mode != selected_mode:
        st.caption(
            f"Requested mode: {MODE_LABELS[selected_mode]} | Effective mode: {MODE_LABELS.get(effective_mode, effective_mode)}"
        )

    if run_pipeline:
        temp_image_path = _write_uploaded_file(uploaded)
        output_key = f"{uploaded.name}|{selected_mode}|{effective_mode}|{layout_model_path}"

        with st.spinner(f"Running {MODE_LABELS.get(effective_mode, effective_mode)}..."):
            try:
                result = _run_pipeline(temp_image_path, effective_mode, cfg)
                result_dict = result.to_dict()
                st.session_state.pipeline_result = result_dict
                st.session_state.pipeline_result_key = output_key
                st.session_state.pipeline_result_error = ""
            except Exception as exc:
                st.session_state.pipeline_result = None
                st.session_state.pipeline_result_key = output_key
                st.session_state.pipeline_result_error = str(exc)

    result_key = st.session_state.get("pipeline_result_key", "")
    output_key = f"{uploaded.name}|{selected_mode}|{effective_mode}|{layout_model_path}"
    pipeline_error = st.session_state.get("pipeline_result_error", "")
    pipeline_result = st.session_state.get("pipeline_result")

    if pipeline_error and result_key == output_key:
        st.error(pipeline_error)

    if pipeline_result and result_key == output_key:
        warnings = pipeline_result.get("metadata", {}).get("warnings", [])
        for warning in warnings:
            st.warning(warning)

        _show_summary(pipeline_result)
        _show_sections(pipeline_result)

        st.download_button(
            "Download Extraction JSON",
            data=json.dumps(pipeline_result, indent=2),
            file_name=f"receipt_{effective_mode}_result.json",
            mime="application/json",
        )
    elif not pipeline_error:
        st.info("Choose a mode and run extraction.")

    st.divider()

    if st.button("Add Current Result To Session History"):
        st.session_state.history.append(
            {
                "file": uploaded.name,
                "requested_mode": MODE_LABELS[selected_mode],
                "effective_mode": MODE_LABELS.get(effective_mode, effective_mode),
                "timestamp": pd.Timestamp.now(),
            }
        )
        st.success("Added to history.")

    if st.session_state.history:
        st.subheader("Session History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download History CSV",
            data=history_df.to_csv(index=False),
            file_name="receipt_ocr_history.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a receipt image to begin extraction.")
