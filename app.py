import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.app.extract_fields import extract_date, extract_totals, guess_merchant
from src.app.extraction_modes import (
    build_easyocr_rules_result,
    build_layoutlmv3_only_result,
    build_hybrid_result,
)
from src.app.receipt_script_parser import parse_receipt_script
from src.app.text_cleaning import clean_ocr_text
from src.evaluate import normalize_text
from src.layoutlmv3_engine import predict_layoutlmv3_from_easyocr, predict_layoutlmv3_with_hybrid
from src.ocr_engine import run_easyocr
from src.preprocess import candidate_modes_for_auto, preprocess_for_ocr
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.runtime.policy import load_runtime_policy


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def ocr_text_from_results(results):
    parts = [r.get("text", "") for r in results if r.get("text")]
    return "\n".join(parts).strip()


def mean_conf(results):
    if not results:
        return 0.0
    return sum(float(r.get("conf", 0.0)) for r in results) / max(len(results), 1)


def text_quality_score(text: str) -> float:
    t = normalize_text(text)
    if not t:
        return 0.0
    alnum = sum(ch.isalnum() for ch in t)
    length = len(t)
    return (alnum / max(length, 1)) + min(length, 80) / 80.0


def blended_score(conf: float, text: str) -> float:
    return (0.6 * conf) + (0.4 * text_quality_score(text))


def run_ocr_on(img_rgb: np.ndarray, mode: str) -> dict:
    processed, _ = preprocess_for_ocr(img_rgb, mode=mode)
    results = run_easyocr(processed)
    text = ocr_text_from_results(results)
    conf = mean_conf(results)
    score = blended_score(conf, text)
    return {"mode": mode, "processed": processed, "results": results, "text": text, "conf": conf, "score": score}


def choose_best_auto(img_rgb: np.ndarray, margin: float = 0.01) -> dict:
    baseline = run_ocr_on(img_rgb, "none")
    best = baseline

    for candidate_mode in candidate_modes_for_auto():
        if candidate_mode == "none":
            continue
        out = run_ocr_on(img_rgb, candidate_mode)
        if out["score"] > best["score"] + margin:
            best = out

    return best


st.set_page_config(page_title="Receipt OCR Extractor", page_icon="🧾", layout="wide")

st.title("🧾 Receipt OCR Extractor")
st.caption("Upload a receipt → OCR → choose extraction mode → export structured results.")

with st.sidebar:
    st.header("OCR Settings")
    mode = st.selectbox("Mode", ["auto", "none", "clahe", "denoise", "otsu", "adaptive"], index=0)
    margin = st.slider("Auto switch margin", 0.00, 0.08, 0.01, 0.01)
    show_processed = st.checkbox("Show processed image", value=True)
    compare_modes = st.checkbox("Compare modes (slow)", value=False)
    show_raw = st.checkbox("Show raw OCR text", value=True)
    show_debug = st.checkbox("Show debug", value=False)

    runtime_policy = load_runtime_policy()

    st.divider()
    default_layout_checkpoint = runtime_policy.preferred_checkpoint or "outputs/layoutlmv3_sroie"
    if default_layout_checkpoint and not Path(default_layout_checkpoint).expanduser().exists():
        default_layout_checkpoint = "outputs/layoutlmv3_sroie"
    if not Path(default_layout_checkpoint).exists():
        default_layout_checkpoint = "microsoft/layoutlmv3-base"

    st.header("LayoutLMv3 Settings")
    layout_model_path = st.text_input(
        "Model/checkpoint path",
        value=default_layout_checkpoint,
        help="Use a fine-tuned receipt KIE checkpoint. Base model is not suitable for extraction.",
    )

    st.divider()
    st.header("Runtime Policy")
    st.caption(f"Default mode: {runtime_policy.default_mode}")
    st.caption(f"Fallback mode: {runtime_policy.fallback_mode_on_model_failure}")
    if runtime_policy.preferred_checkpoint:
        st.caption(f"Preferred checkpoint: {runtime_policy.preferred_checkpoint}")

    st.divider()
    st.header("About")
    st.write("**Auto mode** tries: none / clahe / denoise and picks the best by blended score.")
    st.write("**EasyOCR + Rules**: Rule-based extraction with receipt script parser.")
    st.write("**LayoutLMv3 Only**: Pure semantic entity extraction from model.")
    st.write("**Hybrid**: Combines model output + parser metadata + fusion logic.")


uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)

    # Run OCR once
    if mode == "auto":
        out = choose_best_auto(img_rgb, margin=margin)
    else:
        out = run_ocr_on(img_rgb, mode)

    chosen_mode = out.get("mode", "unknown")
    raw_text = out.get("text") or ""

    try:
        cleaned_text = clean_ocr_text(raw_text)
    except Exception:
        cleaned_text = ""

    cleaned_text = cleaned_text or ""
    if (not cleaned_text.strip()) and raw_text.strip():
        cleaned_text = raw_text

    conf = float(out.get("conf", 0.0))
    score = float(out.get("score", 0.0))
    processed = out.get("processed", None)

    # Display input image and OCR metrics
    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with top_right:
        st.subheader("OCR Summary")
        st.metric("Preprocessing mode", chosen_mode)
        st.metric("Mean confidence", f"{conf:.3f}")
        st.metric("Blended score", f"{score:.3f}")
        if show_processed and processed is not None:
            with st.expander("Show processed image"):
                st.image(processed, caption="Processed for OCR", use_container_width=True)

    st.divider()

    # Mode selection
    st.subheader("Extraction Mode")
    mode_options = ["EasyOCR + Rules", "LayoutLMv3 Only", "Hybrid Extraction"]
    default_mode_label_map = {
        "easyocr_rules": "EasyOCR + Rules",
        "layoutlm_only": "LayoutLMv3 Only",
        "hybrid": "Hybrid Extraction",
    }
    default_mode_label = default_mode_label_map.get(runtime_policy.default_mode, "Hybrid Extraction")
    default_mode_index = mode_options.index(default_mode_label) if default_mode_label in mode_options else 2

    selected_mode = st.radio(
        "Choose extraction method:",
        options=mode_options,
        index=default_mode_index,
        horizontal=True,
    )

    effective_selected_mode = selected_mode
    model_mode_selected = selected_mode in {"LayoutLMv3 Only", "Hybrid Extraction"}
    if model_mode_selected:
        ckpt_path = Path(layout_model_path).expanduser()
        fallback_label = default_mode_label_map.get(runtime_policy.fallback_mode_on_model_failure, "EasyOCR + Rules")
        if not ckpt_path.exists() and layout_model_path != "microsoft/layoutlmv3-base":
            st.warning(
                f"Configured checkpoint is missing: {ckpt_path}. Falling back to {fallback_label}."
            )
            effective_selected_mode = fallback_label
        elif ckpt_path.exists():
            compatibility = inspect_checkpoint_label_space(ckpt_path)
            if (not compatibility.is_compatible) and (not compatibility.is_legacy):
                st.warning(
                    f"Checkpoint incompatible: {compatibility.message} Falling back to {fallback_label}."
                )
                effective_selected_mode = fallback_label
            elif compatibility.is_legacy and not runtime_policy.allow_legacy_checkpoint:
                st.warning(
                    "Checkpoint is legacy/reduced-schema and legacy usage is disabled by runtime policy. "
                    f"Falling back to {fallback_label}."
                )
                effective_selected_mode = fallback_label
            elif compatibility.is_legacy:
                st.info("Legacy checkpoint detected; richer-schema fields may be limited.")

    # ===========================================================================
    # MODE 1: EASYOCR + RULES
    # ===========================================================================
    if effective_selected_mode == "EasyOCR + Rules":
        st.subheader("EasyOCR + Rules Mode")

        # Editable text area for user corrections
        st.markdown("**Extracted text (editable for rule-based corrections):**")
        edited_text = st.text_area("Edit OCR text before parsing:", value=cleaned_text, height=200, key="easy_text")
        edited_text = edited_text or ""

        if show_raw:
            with st.expander("Show raw OCR text"):
                st.text_area("Raw OCR output", value=raw_text, height=150, disabled=True)

        # Parse receipt script
        receipt_script = parse_receipt_script(edited_text)

        # Build mode-specific result
        easyocr_result = build_easyocr_rules_result(
            filename=uploaded.name,
            mode_selected=mode,
            chosen_mode=chosen_mode,
            mean_conf=conf,
            score=score,
            edited_text=edited_text,
            raw_text=raw_text,
            ocr_results=out.get("results", []),
            receipt_script=receipt_script,
            margin=margin if mode == "auto" else None,
        )

        # Display results
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Merchant", easyocr_result["header"].get("merchant") or "-")
        col2.metric("Date", easyocr_result["header"].get("date") or "-")
        col3.metric("Invoice #", easyocr_result["header"].get("invoice_no") or "-")
        col4.metric("GST ID", easyocr_result["header"].get("gst_id") or "-")

        with st.expander("📋 Header Info"):
            st.json(easyocr_result["header"])

        with st.expander("💰 Totals"):
            st.json(easyocr_result["totals"])

        with st.expander("📦 Line Items"):
            if easyocr_result["line_items"]:
                st.dataframe(pd.DataFrame(easyocr_result["line_items"]), use_container_width=True)
            else:
                st.info("No line items extracted")

        with st.expander("🏷️ Tax Summary"):
            if easyocr_result["tax_summary"]:
                st.json(easyocr_result["tax_summary"])
            else:
                st.info("No tax summary found")

        with st.expander("📄 Script Lines (Raw OCR)"):
            st.text("\n".join(easyocr_result["script_lines"]))

        # Download button
        st.download_button(
            "📥 Download EasyOCR + Rules JSON",
            data=json.dumps(to_jsonable(easyocr_result), indent=2),
            file_name="receipt_easyocr_rules_result.json",
            mime="application/json",
        )

    # ===========================================================================
    # MODE 2: LAYOUTLMV3 ONLY
    # ===========================================================================
    elif effective_selected_mode == "LayoutLMv3 Only":
        st.subheader("LayoutLMv3 Only Mode (Pure Model)")

        run_layout = st.button("🚀 Run LayoutLMv3 Inference", key="run_layoutlmv3_only")
        output_key = f"layoutlmv3_only|{uploaded.name}|{mode}|{chosen_mode}|{layout_model_path}"

        if run_layout:
            with st.spinner("Running LayoutLMv3 inference (model-only)..."):
                try:
                    prediction = predict_layoutlmv3_from_easyocr(
                        image_rgb=img_rgb,
                        ocr_results=out.get("results", []),
                        model_name_or_path=layout_model_path,
                    )
                    st.session_state.layoutlmv3_only_pred = prediction
                    st.session_state.layoutlmv3_only_key = output_key
                    st.session_state.layoutlmv3_only_error = ""
                except Exception as exc:
                    st.session_state.layoutlmv3_only_pred = None
                    st.session_state.layoutlmv3_only_key = output_key
                    st.session_state.layoutlmv3_only_error = str(exc)

        prediction = st.session_state.get("layoutlmv3_only_pred")
        prediction_key = st.session_state.get("layoutlmv3_only_key", "")
        prediction_error = st.session_state.get("layoutlmv3_only_error", "")

        if prediction_error and prediction_key == output_key:
            st.error(f"Error: {prediction_error}")

        if prediction and prediction_key == output_key:
            prediction_warning = prediction.get("warning", "")
            if prediction_warning:
                st.warning(prediction_warning)
            else:
                if prediction.get("was_truncated"):
                    st.warning("⚠️ OCR tokens were truncated to 512 words before inference.")

                # Build mode-specific result
                layoutlmv3_result = build_layoutlmv3_only_result(
                    filename=uploaded.name,
                    model_path=layout_model_path,
                    mode_selected=mode,
                    chosen_mode=chosen_mode,
                    prediction=prediction,
                )

                # Display metrics
                fields = layoutlmv3_result["fields"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Merchant", fields.get("merchant") or "-")
                col2.metric("Date", fields.get("date") or "-")
                col3.metric("Address", fields.get("address") or "-")
                col4.metric("Total", fields.get("total") or "-")

                # Display entities
                with st.expander("🏷️ Predicted Entities"):
                    entities = layoutlmv3_result["raw_entities"]
                    if entities:
                        st.dataframe(pd.DataFrame(entities), use_container_width=True)
                    else:
                        st.info("No entities predicted")

                with st.expander("📊 Grouped Entities"):
                    grouped = layoutlmv3_result.get("grouped_entities", [])
                    if grouped:
                        st.json(grouped)
                    else:
                        st.info("No grouped entities")

                # Download button
                st.download_button(
                    "📥 Download LayoutLMv3 Only JSON",
                    data=json.dumps(to_jsonable(layoutlmv3_result), indent=2),
                    file_name="receipt_layoutlmv3_only_result.json",
                    mime="application/json",
                )
        else:
            st.info("👆 Click the button above to run model inference")

    # ===========================================================================
    # MODE 3: HYBRID
    # ===========================================================================
    elif effective_selected_mode == "Hybrid Extraction":
        st.subheader("Hybrid Mode (Model + Parser + Fusion)")

        run_hybrid = st.button("🚀 Run Hybrid Extraction", key="run_hybrid")
        output_key = f"hybrid|{uploaded.name}|{mode}|{chosen_mode}|{layout_model_path}"

        if run_hybrid:
            with st.spinner("Running hybrid extraction (model + parser + fusion)..."):
                try:
                    # Get model prediction and OCR text
                    model_pred, ocr_text = predict_layoutlmv3_with_hybrid(
                        image_rgb=img_rgb,
                        ocr_results=out.get("results", []),
                        model_name_or_path=layout_model_path,
                    )

                    # Parse receipt script
                    parser_output = parse_receipt_script(ocr_text)

                    # Build hybrid result
                    hybrid_result = build_hybrid_result(
                        filename=uploaded.name,
                        model_path=layout_model_path,
                        mode_selected=mode,
                        chosen_mode=chosen_mode,
                        model_prediction=model_pred,
                        parser_output=parser_output,
                        ocr_text=ocr_text,
                    )

                    st.session_state.hybrid_result = hybrid_result
                    st.session_state.hybrid_key = output_key
                    st.session_state.hybrid_error = ""
                except Exception as exc:
                    st.session_state.hybrid_result = None
                    st.session_state.hybrid_key = output_key
                    st.session_state.hybrid_error = str(exc)

        hybrid_result = st.session_state.get("hybrid_result")
        hybrid_key = st.session_state.get("hybrid_key", "")
        hybrid_error = st.session_state.get("hybrid_error", "")

        if hybrid_error and hybrid_key == output_key:
            st.error(f"Error: {hybrid_error}")

        if hybrid_result and hybrid_key == output_key:
            warnings = hybrid_result.get("warnings", [])
            if warnings and warnings[0]:
                st.warning(warnings[0])
            else:
                # Display best fused output
                st.markdown("### 🎯 Best Fused Output")
                fused = hybrid_result.get("final_fused", {})

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Merchant", fused.get("header", {}).get("merchant") or "-")
                col2.metric("Date", fused.get("header", {}).get("date") or "-")
                col3.metric("Invoice #", fused.get("header", {}).get("invoice_no") or "-")
                col4.metric("Payable Amount", fused.get("totals", {}).get("total_amt_payable") or "-")

                # Display model component
                with st.expander("🤖 Model Output (LayoutLMv3)"):
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    model_fields = hybrid_result.get("fields", {})
                    col_m1.metric("Model: Merchant", model_fields.get("merchant") or "-")
                    col_m2.metric("Model: Date", model_fields.get("date") or "-")
                    col_m3.metric("Model: Address", model_fields.get("address") or "-")
                    col_m4.metric("Model: Total", model_fields.get("total") or "-")

                    st.markdown("**Raw Entities**")
                    entities = hybrid_result.get("raw_entities", [])
                    if entities:
                        st.dataframe(pd.DataFrame(entities), use_container_width=True)
                    else:
                        st.info("No entities predicted")

                # Display parser component
                with st.expander("📋 Parser Output (Receipt Script)"):
                    parser = hybrid_result.get("receipt_script", {})

                    st.markdown("**Header**")
                    st.json(parser.get("header", {}))

                    st.markdown("**Totals**")
                    st.json(parser.get("totals", {}))

                    st.markdown("**Line Items**")
                    if parser.get("line_items"):
                        st.dataframe(pd.DataFrame(parser.get("line_items", [])), use_container_width=True)
                    else:
                        st.info("No line items found")

                    st.markdown("**Tax Summary**")
                    if parser.get("tax_summary"):
                        st.json(parser.get("tax_summary", []))
                    else:
                        st.info("No tax summary found")

                # Display fused output
                with st.expander("✨ Fused Output (Best Merged)"):
                    st.json(fused)

                # Download button
                st.download_button(
                    "📥 Download Hybrid Result JSON",
                    data=json.dumps(to_jsonable(hybrid_result), indent=2),
                    file_name="receipt_hybrid_result.json",
                    mime="application/json",
                )
        else:
            st.info("👆 Click the button above to run hybrid extraction")

    st.divider()

    # Session history (works for all modes)
    if st.button("➕ Add current result to session history"):
        st.session_state.history.append(
            {
                "file": uploaded.name,
                "mode": selected_mode,
                "effective_mode": effective_selected_mode,
                "timestamp": pd.Timestamp.now(),
            }
        )
        st.success("Added to history!")

    if st.session_state.history:
        st.subheader("📊 Session History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        if st.button("📥 Download History CSV"):
            st.download_button(
                "Download CSV",
                data=history_df.to_csv(index=False),
                file_name="receipt_ocr_history.csv",
                mime="text/csv",
            )

else:
    st.info("📤 Upload a receipt image to begin extraction.")

