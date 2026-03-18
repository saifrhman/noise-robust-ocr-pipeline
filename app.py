import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.receipt_ai.runtime.output import format_result_output
from src.receipt_ai.runtime.runner import MODE_LABELS, load_runtime_config, resolve_mode, run_extraction

LABEL_TO_MODE = {label: mode for mode, label in MODE_LABELS.items()}


def _write_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getvalue())
        return Path(handle.name)


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

cfg, runtime_policy = load_runtime_config()

with st.sidebar:
    st.header("Default Config")
    st.caption(f"Default mode: {runtime_policy.default_mode}")
    st.caption(f"Fallback mode: {runtime_policy.fallback_mode_on_model_failure}")
    if runtime_policy.preferred_checkpoint:
        st.caption(f"Preferred checkpoint: {runtime_policy.preferred_checkpoint}")
    if runtime_policy.decision:
        st.caption(f"Promotion decision: {runtime_policy.decision.get('status', 'n/a')}")

    st.divider()
    st.header("LayoutLMv3 Settings")
    default_checkpoint = runtime_policy.preferred_checkpoint or str(cfg.paths.model_checkpoint or "")
    layout_model_path = st.text_input(
        "Fine-tuned checkpoint path",
        value=default_checkpoint,
        help="Required for LayoutLMv3 Only and Hybrid modes. Base pretrained checkpoints are not supported.",
    ).strip()

    st.divider()
    st.header("Output")
    default_output_mode = str(runtime_policy.output.get("mode", "full"))
    output_mode = st.radio(
        "JSON output mode",
        options=["full", "minimal"],
        index=0 if default_output_mode != "minimal" else 1,
        horizontal=True,
    )
    include_confidence = st.checkbox("Include confidence", value=bool(runtime_policy.output.get("include_confidence", True)))
    include_provenance = st.checkbox("Include provenance", value=bool(runtime_policy.output.get("include_provenance", True)))

    st.divider()
    st.header("About")
    st.write("The app uses the unified `src/receipt_ai` pipeline and `default_config.json` by default.")
    st.write("Manual overrides here affect only the current UI session.")

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

    effective_mode, checkpoint_used, mode_messages = resolve_mode(
        selected_mode,
        checkpoint_override=layout_model_path,
        cfg=cfg,
        policy=runtime_policy,
    )
    for message in mode_messages:
        if "falling back" in message.lower():
            st.warning(message)
        else:
            st.info(message)

    st.caption(f"Selected mode: {MODE_LABELS[selected_mode]}")
    st.caption(f"Effective mode: {MODE_LABELS.get(effective_mode, effective_mode)}")
    st.caption(f"Checkpoint used: {checkpoint_used or 'none'}")
    st.caption(f"Fallback behavior: {runtime_policy.fallback_mode_on_model_failure}")

    if run_pipeline:
        temp_image_path = _write_uploaded_file(uploaded)
        output_key = f"{uploaded.name}|{selected_mode}|{effective_mode}|{layout_model_path}|{output_mode}|{int(include_confidence)}|{int(include_provenance)}"

        with st.spinner(f"Running {MODE_LABELS.get(effective_mode, effective_mode)}..."):
            try:
                result = run_extraction(temp_image_path, mode=effective_mode, cfg=cfg)
                result_dict = format_result_output(
                    result,
                    output_mode=output_mode,
                    include_confidence=include_confidence,
                    include_provenance=include_provenance,
                )
                st.session_state.pipeline_result = result_dict
                st.session_state.pipeline_result_key = output_key
                st.session_state.pipeline_result_error = ""
            except Exception as exc:
                st.session_state.pipeline_result = None
                st.session_state.pipeline_result_key = output_key
                st.session_state.pipeline_result_error = str(exc)

    result_key = st.session_state.get("pipeline_result_key", "")
    output_key = f"{uploaded.name}|{selected_mode}|{effective_mode}|{layout_model_path}|{output_mode}|{int(include_confidence)}|{int(include_provenance)}"
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
