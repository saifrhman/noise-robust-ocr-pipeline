import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.app.extract_fields import extract_date, extract_totals, guess_merchant
from src.app.text_cleaning import clean_ocr_text
from src.evaluate import normalize_text
from src.layoutlmv3_engine import predict_layoutlmv3_from_easyocr
from src.ocr_engine import run_easyocr
from src.preprocess import candidate_modes_for_auto, preprocess_for_ocr


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
st.caption("Upload a receipt -> OCR with adaptive preprocessing -> extract totals/dates -> export results.")

with st.sidebar:
    st.header("OCR Settings")
    mode = st.selectbox("Mode", ["auto", "none", "clahe", "denoise", "otsu", "adaptive"], index=0)
    margin = st.slider("Auto switch margin", 0.00, 0.08, 0.01, 0.01)
    show_processed = st.checkbox("Show processed image", value=True)
    compare_modes = st.checkbox("Compare modes (slow)", value=False)
    show_raw = st.checkbox("Show raw OCR text", value=True)
    show_debug = st.checkbox("Show debug", value=False)

    st.divider()
    st.header("LayoutLMv3")
    layout_model_path = st.text_input(
        "Model/checkpoint path",
        value="microsoft/layoutlmv3-base",
        help="Use a fine-tuned receipt KIE checkpoint. Base model is not suitable for extraction.",
    )

    st.divider()
    st.header("About")
    st.write("Auto mode tries: none / clahe / denoise and picks the best by a blended score.")
    st.write("LayoutLMv3 uses OCR tokens and bounding boxes to predict structured entities.")


uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)

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

    parse_text = raw_text
    conf = float(out.get("conf", 0.0))
    score = float(out.get("score", 0.0))
    processed = out.get("processed", None)

    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with top_right:
        st.subheader("OCR Output")
        st.markdown(f"**Chosen mode:** `{chosen_mode}`")
        st.markdown(f"**Mean confidence:** `{conf:.3f}`")
        st.markdown(f"**Blended score:** `{score:.3f}`")
        if show_processed and processed is not None:
            st.image(processed, caption="Processed for OCR", use_container_width=True)

    st.divider()

    easy_tab, layout_tab = st.tabs(["EasyOCR + Rules", "LayoutLMv3"])

    with easy_tab:
        st.subheader("Extracted Text (editable)")
        edited_text = st.text_area("Edit before extracting fields/export:", value=cleaned_text, height=240)
        edited_text = edited_text or ""

        if show_raw:
            with st.expander("Show raw OCR text"):
                st.text_area("Raw OCR", value=raw_text, height=180)

        if show_debug:
            with st.expander("Debug: lengths and parsing source"):
                st.write("raw length:", len(raw_text))
                st.write("cleaned length:", len(cleaned_text))
                st.write("edited length:", len(edited_text))
                st.write("Parsing fields from:", "raw_text")

        merchant = guess_merchant(parse_text)
        date = extract_date(parse_text)
        totals = extract_totals(parse_text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Merchant (guess)", merchant or "-")
        c2.metric("Date (first match)", date or "-")
        c3.metric("Total candidates", ", ".join(totals[:3]) if totals else "-")

        result = {
            "file": uploaded.name,
            "mode_selected": mode,
            "chosen_mode": chosen_mode,
            "auto_margin": margin if mode == "auto" else None,
            "mean_conf": float(conf),
            "score": float(score),
            "merchant_guess": merchant,
            "date": date,
            "totals": totals,
            "text": edited_text,
        }

        if compare_modes:
            st.subheader("Mode Comparison")
            rows = []
            for candidate_mode in ["none", "clahe", "denoise"]:
                candidate_out = run_ocr_on(img_rgb, candidate_mode)
                rows.append(
                    {
                        "mode": candidate_mode,
                        "conf": float(candidate_out.get("conf", 0.0)),
                        "score": float(candidate_out.get("score", 0.0)),
                        "preview": (
                            candidate_out.get("text", "")[:80]
                            + ("..." if len(candidate_out.get("text", "")) > 80 else "")
                        ),
                    }
                )
            compare_df = pd.DataFrame(rows).sort_values("score", ascending=False)
            st.dataframe(compare_df, use_container_width=True)

        a1, a2, _ = st.columns([1, 1, 2])
        with a1:
            if st.button("Add to session history"):
                st.session_state.history.append(result)
                st.success("Added to history.")
        with a2:
            st.download_button(
                "Download JSON",
                data=pd.Series(result).to_json(),
                file_name="receipt_ocr_result.json",
                mime="application/json",
            )

        if st.session_state.history:
            st.subheader("Session History")
            history_df = pd.DataFrame(
                [
                    {
                        "file": row["file"],
                        "chosen_mode": row["chosen_mode"],
                        "conf": row["mean_conf"],
                        "score": row["score"],
                        "merchant": row["merchant_guess"],
                        "date": row["date"],
                        "totals": ", ".join((row["totals"] or [])[:2]) if row.get("totals") else "",
                    }
                    for row in st.session_state.history
                ]
            )
            st.dataframe(history_df, use_container_width=True)

            st.download_button(
                "Download History CSV",
                data=history_df.to_csv(index=False),
                file_name="receipt_ocr_history.csv",
                mime="text/csv",
            )

    with layout_tab:
        st.subheader("LayoutLMv3 Output")
        st.caption("Run a fine-tuned checkpoint to get merchant/date/address/total entities.")

        run_layout = st.button("Run LayoutLMv3", key="run_layoutlmv3")
        output_key = f"{uploaded.name}|{mode}|{chosen_mode}|{layout_model_path}"

        if run_layout:
            with st.spinner("Running LayoutLMv3 inference..."):
                try:
                    prediction = predict_layoutlmv3_from_easyocr(
                        image_rgb=img_rgb,
                        ocr_results=out.get("results", []),
                        model_name_or_path=layout_model_path,
                    )
                    st.session_state.layoutlmv3_prediction = prediction
                    st.session_state.layoutlmv3_prediction_key = output_key
                    st.session_state.layoutlmv3_error = ""
                except Exception as exc:
                    st.session_state.layoutlmv3_prediction = None
                    st.session_state.layoutlmv3_prediction_key = output_key
                    st.session_state.layoutlmv3_error = str(exc)

        prediction = st.session_state.get("layoutlmv3_prediction")
        prediction_key = st.session_state.get("layoutlmv3_prediction_key", "")
        prediction_error = st.session_state.get("layoutlmv3_error", "")

        if prediction_error and prediction_key == output_key:
            st.error(prediction_error)

        if prediction and prediction_key == output_key:
            prediction_warning = prediction.get("warning", "")
            if prediction_warning:
                st.warning(prediction_warning)

            if prediction.get("was_truncated"):
                st.warning("OCR tokens were truncated to 512 words before inference.")

            fields = prediction.get("fields", {})
            merchant_pred = fields.get("merchant") or None
            date_pred = fields.get("date") or None
            address_pred = fields.get("address") or None
            total_pred = fields.get("total") or None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Merchant (LayoutLMv3)", merchant_pred or "-")
            c2.metric("Date (LayoutLMv3)", date_pred or "-")
            c3.metric("Address (LayoutLMv3)", address_pred or "-")
            c4.metric("Total (LayoutLMv3)", total_pred or "-")

            entities = prediction.get("raw_entities", prediction.get("entities", []))
            if entities:
                st.markdown("**Predicted Entities**")
                st.dataframe(pd.DataFrame(entities), use_container_width=True)
            else:
                st.info("No entities predicted by this checkpoint.")

            payload = {
                "file": uploaded.name,
                "model": layout_model_path,
                "mode_selected": mode,
                "chosen_mode": chosen_mode,
                "num_words": prediction.get("num_words", 0),
                "fields": fields,
                "entities": entities,
            }
            st.download_button(
                "Download LayoutLMv3 JSON",
                data=json.dumps(payload, indent=2),
                file_name="receipt_layoutlmv3_result.json",
                mime="application/json",
            )
        else:
            st.info("Click 'Run LayoutLMv3' to generate output in this tab.")
else:
    st.info("Upload a receipt image to begin.")
