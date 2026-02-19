# app.py
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from src.preprocess import preprocess_for_ocr, candidate_modes_for_auto
from src.ocr_engine import run_easyocr
from src.evaluate import normalize_text

from src.app.extract_fields import extract_date, extract_totals, guess_merchant
from src.app.text_cleaning import clean_ocr_text


# -----------------------------
# Helpers (self-contained)
# -----------------------------
def ocr_text_from_results(results):
    parts = [r.get("text", "") for r in results if r.get("text")]
    # Keep line breaks (better for receipts); if OCR results are already per-line, this helps.
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

    for m in candidate_modes_for_auto():
        if m == "none":
            continue
        out = run_ocr_on(img_rgb, m)
        if out["score"] > best["score"] + margin:
            best = out

    return best


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Receipt OCR Extractor", page_icon="🧾", layout="wide")

st.title("🧾 Receipt OCR Extractor")
st.caption("Upload a receipt → OCR with adaptive preprocessing → extract totals/dates → export results.")

with st.sidebar:
    st.header("OCR Settings")
    mode = st.selectbox("Mode", ["auto", "none", "clahe", "denoise", "otsu", "adaptive"], index=0)
    margin = st.slider("Auto switch margin", 0.00, 0.08, 0.01, 0.01)
    show_processed = st.checkbox("Show processed image", value=True)
    compare_modes = st.checkbox("Compare modes (slow)", value=False)
    show_raw = st.checkbox("Show raw OCR text", value=True)
    show_debug = st.checkbox("Show debug", value=False)

    st.divider()
    st.header("About")
    st.write("Auto mode tries: none / clahe / denoise and picks the best by a blended score.")


uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)

    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    # Run OCR based on chosen mode
    if mode == "auto":
        out = choose_best_auto(img_rgb, margin=margin)
    else:
        out = run_ocr_on(img_rgb, mode)

    chosen_mode = out.get("mode", "unknown")

    raw_text = out.get("text") or ""

    # Cleaning can occasionally be overly aggressive / fail.
    # Always keep a safe fallback so the app never shows an empty box if OCR found text.
    try:
        cleaned_text = clean_ocr_text(raw_text)
    except Exception:
        cleaned_text = ""

    cleaned_text = cleaned_text or ""
    raw_text = raw_text or ""

    # ✅ Fallback: never show empty if OCR produced text
    if (not cleaned_text.strip()) and raw_text.strip():
        cleaned_text = raw_text

    # ✅ IMPORTANT: Use RAW text for field extraction (structure),
    # while using CLEANED text for display/editing.
    parse_text = raw_text

    conf = float(out.get("conf", 0.0))
    score = float(out.get("score", 0.0))
    processed = out.get("processed", None)

    with top_right:
        st.subheader("OCR Output")
        st.markdown(f"**Chosen mode:** `{chosen_mode}`")
        st.markdown(f"**Mean confidence:** `{conf:.3f}`")
        st.markdown(f"**Blended score:** `{score:.3f}`")

        if show_processed and processed is not None:
            st.image(processed, caption="Processed for OCR", use_container_width=True)

    st.divider()

    # Editable text area uses CLEANED text (with fallback to raw)
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

    # ✅ Field extraction uses parse_text (raw) for better structure
    merchant = guess_merchant(parse_text)
    date = extract_date(parse_text)
    totals = extract_totals(parse_text)

    c1, c2, c3 = st.columns(3)
    c1.metric("Merchant (guess)", merchant or "—")
    c2.metric("Date (first match)", date or "—")
    c3.metric("Total candidates", ", ".join(totals[:3]) if totals else "—")

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
        # export the editable (cleaned) text the user sees/edits
        "text": edited_text,
    }

    # Compare modes table (optional)
    if compare_modes:
        st.subheader("Mode Comparison")
        rows = []
        for m in ["none", "clahe", "denoise"]:
            o = run_ocr_on(img_rgb, m)
            rows.append(
                {
                    "mode": m,
                    "conf": float(o.get("conf", 0.0)),
                    "score": float(o.get("score", 0.0)),
                    "preview": (o.get("text", "")[:80] + ("..." if len(o.get("text", "")) > 80 else "")),
                }
            )
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        st.dataframe(df, use_container_width=True)

    # Actions
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

    # History table + CSV export
    if st.session_state.history:
        st.subheader("Session History")
        dfh = pd.DataFrame(
            [
                {
                    "file": h["file"],
                    "chosen_mode": h["chosen_mode"],
                    "conf": h["mean_conf"],
                    "score": h["score"],
                    "merchant": h["merchant_guess"],
                    "date": h["date"],
                    "totals": ", ".join((h["totals"] or [])[:2]) if h.get("totals") else "",
                }
                for h in st.session_state.history
            ]
        )
        st.dataframe(dfh, use_container_width=True)

        st.download_button(
            "Download History CSV",
            data=dfh.to_csv(index=False),
            file_name="receipt_ocr_history.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a receipt image to begin.")
