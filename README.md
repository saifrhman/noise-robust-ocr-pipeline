# Noise-Robust OCR Pipeline

Receipt OCR pipeline focused on noisy real-world images.

The project includes:
- EasyOCR with adaptive preprocessing (`none`, `clahe`, `denoise`)
- Heuristic receipt field extraction (merchant/date/total)
- SROIE evaluation scripts
- Streamlit app with two output tabs:
  - `EasyOCR + Rules`
  - `LayoutLMv3`

## Features

- EasyOCR text detection and recognition
- Auto mode that selects the best preprocessing strategy
- OCR confidence + text-quality blended scoring
- Streamlit UI for upload, extraction, JSON/CSV export
- LayoutLMv3 integration for token-level entity extraction
- LayoutLMv3 fine-tuning pipeline for receipt KIE with semantic BIO labels

## Project Structure

```text
app.py
train_layoutlmv3_receipts.py
infer_layoutlmv3_receipt.py
data_utils.py
label_config.py
README_finetuning.md
scripts/
  train_layoutlmv3.py
src/
  layoutlmv3_engine.py
  ocr_engine.py
  preprocess.py
  run_sroie_eval.py
  app/
    extract_fields.py
    text_cleaning.py
```

## Setup

```bash
git clone https://github.com/saifrhman/noise-robust-ocr-pipeline.git
cd noise-robust-ocr-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

In the app:
- Use `EasyOCR + Rules` tab for current heuristic extraction.
- Use `LayoutLMv3` tab and click `Run LayoutLMv3` for model-based entity output.

Important:
- `microsoft/layoutlmv3-base` is not fine-tuned for receipt entities.
- For meaningful merchant/date/total extraction, use a fine-tuned checkpoint path in the sidebar.

For the complete SROIE fine-tuning workflow, see `README_finetuning.md`.

## Fine-Tune LayoutLMv3

Fine-tuning script:
- `scripts/train_layoutlmv3.py`

Expected JSONL format (`train.jsonl` and `val.jsonl`):

```json
{"id":"sample-1","image_path":"data/receipts/img/1.jpg","tokens":["ACME","STORE"],"bboxes":[[10,20,120,45],[125,20,220,45]],"labels":["B-VENDOR","I-VENDOR"]}
```

Rules:
- `tokens`, `bboxes`, and `labels` must have the same length.
- Bounding boxes should be `[x1, y1, x2, y2]`.
- The training script accepts either normalized (`0..1000`) boxes or pixel boxes and normalizes pixel boxes automatically.
- Labels should use BIO format where possible (for example `B-DATE`, `I-DATE`, `B-TOTAL`).

Training command example:

```bash
python scripts/train_layoutlmv3.py \
  --train-jsonl data/layoutlmv3/train.jsonl \
  --val-jsonl data/layoutlmv3/val.jsonl \
  --output-dir outputs/layoutlmv3_receipts \
  --model-name microsoft/layoutlmv3-base \
  --epochs 8 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --learning-rate 3e-5
```

After training:
- Set sidebar `Model/checkpoint path` to `outputs/layoutlmv3_receipts`
- Open `LayoutLMv3` tab and run inference.

## Evaluate EasyOCR Pipeline on SROIE

```bash
python -m src.run_sroie_eval --split train --mode auto --max 200
python -m src.run_sroie_eval --split test --mode auto --max 200
```

## Notes

- The EasyOCR pipeline remains the baseline and fallback.
- LayoutLMv3 quality depends strongly on the dataset label quality and coverage.
