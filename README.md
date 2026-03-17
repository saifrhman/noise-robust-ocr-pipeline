# Noise-Robust OCR Pipeline

Receipt extraction project built around EasyOCR, LayoutLMv3, a rule-based receipt parser, and a Streamlit frontend.

The application supports three distinct extraction modes:

1. `EasyOCR + Rules`
2. `LayoutLMv3 Only`
3. `Hybrid Extraction`

Each mode runs a separate pipeline and returns a different JSON schema by design.

## Overview

This project is designed for noisy real-world receipt images where OCR output is imperfect and extraction quality depends on a combination of preprocessing, text cleanup, structured parsing, and model-based field detection.

Core capabilities:

- EasyOCR-based text extraction
- OCR preprocessing for noisy receipt images
- automatic preprocessing selection based on OCR quality
- rule-based receipt parsing from OCR script text
- LayoutLMv3 token classification for semantic receipt fields
- hybrid fusion of parser output and model output
- Streamlit UI for upload, review, and JSON export
- training and evaluation utilities for receipt extraction workflows

## Extraction Modes

### EasyOCR + Rules

Purpose:
- transparent rule-based baseline extraction

Pipeline:
- preprocess image
- run EasyOCR
- group OCR text into script lines
- parse receipt structure using rules
- normalize values

Output schema:

```json
{
  "mode": "easyocr_rules",
  "metadata": {"file": "receipt.jpg"},
  "header": {},
  "totals": {},
  "line_items": [],
  "tax_summary": [],
  "script_lines": [],
  "raw_ocr_results": []
}
```

Notes:
- does not include LayoutLMv3 fields
- does not include `raw_entities`
- does not include hybrid fusion output

### LayoutLMv3 Only

Purpose:
- pure model evaluation

Pipeline:
- preprocess image
- run EasyOCR to obtain OCR words and boxes
- run LayoutLMv3 token classification
- group semantic entities
- normalize model output only

Output schema:

```json
{
  "mode": "layoutlmv3_only",
  "metadata": {"file": "receipt.jpg"},
  "fields": {},
  "raw_entities": [],
  "warnings": [],
  "grouped_entities": []
}
```

Notes:
- does not include `receipt_script`
- does not include parser-derived header, totals, or line items
- does not fill missing values from rules

### Hybrid Extraction

Purpose:
- best-quality production-oriented extraction

Pipeline:
- preprocess image
- run EasyOCR
- run LayoutLMv3 inference
- parse receipt script from OCR text
- normalize both outputs
- fuse parser and model evidence into a final result

Output schema:

```json
{
  "mode": "hybrid",
  "metadata": {"file": "receipt.jpg"},
  "fields": {},
  "raw_entities": [],
  "receipt_script": {
    "header": {},
    "totals": {},
    "line_items": [],
    "tax_summary": [],
    "script_lines": []
  },
  "final_fused": {
    "header": {},
    "totals": {},
    "line_items": [],
    "tax_summary": []
  },
  "warnings": []
}
```

Notes:
- retains both model output and parser output
- adds `final_fused` as the best merged result
- intended as the main downstream extraction mode

## Project Architecture

### OCR Layer

- `src/ocr_engine.py`
  Runs EasyOCR and returns text, confidence values, and OCR token data.

- `src/preprocess.py`
  Applies preprocessing strategies such as `none`, `clahe`, `denoise`, `otsu`, and `adaptive`.

### Rule-Based Parsing Layer

- `src/app/receipt_script_parser.py`
  Parses OCR text into structured receipt data:
  - header
  - totals
  - line items
  - tax summary
  - script lines

- `src/app/extract_fields.py`
  Provides lightweight helper extraction for simple merchant, date, and total heuristics.

- `src/app/text_cleaning.py`
  Applies OCR cleanup before parsing.

### Model Layer

- `src/layoutlmv3_engine.py`
  Handles LayoutLMv3 inference from OCR words and bounding boxes.

This module provides:
- pure model inference for `LayoutLMv3 Only`
- hybrid helper flow for `Hybrid Extraction`

### Mode and Fusion Layer

- `src/app/extraction_modes.py`
  Centralizes:
  - normalization helpers
  - amount/date cleanup
  - invoice number normalization
  - mode-specific JSON builders
  - hybrid fusion logic

### UI Layer

- `app.py`
  Streamlit application with explicit mode selection and mode-specific rendering/export.

## Project Structure

```text
app.py
infer_layoutlmv3_receipt.py
label_config.py
packages.txt
README.md
requirements.txt
train_layoutlmv3_receipts.py
assets/
  lexicon_dictionary.txt
outputs/
scripts/
  build_lexicon.py
  train_layoutlmv3.py
src/
  __init__.py
  evaluate.py
  layoutlmv3_engine.py
  main.py
  ocr_engine.py
  preprocess.py
  run_sroie_eval.py
  app/
    extract_fields.py
    extraction_modes.py
    receipt_script_parser.py
    text_cleaning.py
```

## Setup

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### System packages

Some deployments may also use `packages.txt` for system-level dependencies.

## Main Dependencies

- `easyocr`
- `opencv-python-headless`
- `numpy`
- `pandas`
- `streamlit`
- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `symspellpy`

## Running the Streamlit App

```bash
streamlit run app.py
```

The app provides:

- image upload
- OCR preprocessing selection
- auto mode for selecting the best preprocessing result
- checkpoint selection for LayoutLMv3
- explicit selection between:
  - `EasyOCR + Rules`
  - `LayoutLMv3 Only`
  - `Hybrid Extraction`

### Streamlit outputs

#### EasyOCR + Rules

Shows:
- header
- totals
- line items
- tax summary
- script lines

Download:
- `receipt_easyocr_rules_result.json`

#### LayoutLMv3 Only

Shows:
- normalized model fields
- raw entities
- grouped entities
- warnings

Download:
- `receipt_layoutlmv3_only_result.json`

#### Hybrid Extraction

Shows:
- model output
- parser output
- fused output

Download:
- `receipt_hybrid_result.json`

## Command-Line Inference

Main CLI entry point:

- `infer_layoutlmv3_receipt.py`

### LayoutLMv3 Only

```bash
python infer_layoutlmv3_receipt.py \
  --image path/to/receipt.jpg \
  --checkpoint outputs/layoutlmv3_sroie \
  --extraction-mode layoutlmv3_only \
  --pretty
```

### EasyOCR + Rules

```bash
python infer_layoutlmv3_receipt.py \
  --image path/to/receipt.jpg \
  --checkpoint outputs/layoutlmv3_sroie \
  --extraction-mode easyocr_rules \
  --pretty
```

### Hybrid

```bash
python infer_layoutlmv3_receipt.py \
  --image path/to/receipt.jpg \
  --checkpoint outputs/layoutlmv3_sroie \
  --extraction-mode hybrid \
  --pretty
```

Notes:
- `--checkpoint` is still required for model-only and hybrid flows
- `microsoft/layoutlmv3-base` is not a receipt extraction checkpoint
- for meaningful entity extraction, use a fine-tuned checkpoint

## Preprocessing Modes

Supported preprocessing modes:

- `auto`
- `none`
- `clahe`
- `denoise`
- `otsu`
- `adaptive`

`auto` evaluates candidate preprocessing results and picks the best output using a blended OCR score.

## Normalization and Cleanup

Cleanup is implemented in code rather than as manual post-processing.

Implemented normalization includes:

- date normalization to `YYYY-MM-DD` where possible
- amount normalization to 2 decimal places
- invoice number normalization for common OCR variants
- obvious OCR cleanup for frequent receipt text mistakes

Hybrid mode additionally uses fusion logic to prefer values supported by:

1. agreement between parser and model
2. OCR evidence
3. arithmetic consistency
4. parser structure for totals and line items
5. model semantics for fields the parser missed

## LayoutLMv3 Training

Primary fine-tuning script:

- `train_layoutlmv3_receipts.py`

Example:

```bash
python train_layoutlmv3_receipts.py \
  --train-dir data/SROIE2019/train \
  --val-dir data/SROIE2019/test \
  --output-dir outputs/layoutlmv3_sroie
```

Legacy JSONL training script:

- `scripts/train_layoutlmv3.py`

Example:

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

1. Point the Streamlit sidebar checkpoint path to the trained model directory.
2. Use `LayoutLMv3 Only` to inspect pure model quality.
3. Use `Hybrid Extraction` to evaluate the fused production-style output.

## Evaluation

EasyOCR pipeline evaluation:

```bash
python -m src.run_sroie_eval --split train --mode auto --max 200
python -m src.run_sroie_eval --split test --mode auto --max 200
```

## Design Principles

- EasyOCR remains the OCR backbone for all modes
- `LayoutLMv3 Only` stays pure and model-driven
- `Hybrid Extraction` is the only mode that intentionally combines parser and model output
- JSON schemas are explicit and mode-specific
- cleanup and normalization happen inside the application code

## Limitations

- `microsoft/layoutlmv3-base` is not sufficient for useful receipt extraction without fine-tuning
- OCR quality strongly affects both parser and model performance
- tax and line-item extraction depend on OCR preserving enough structure
- hybrid fusion is rule-guided and can still be improved further

## Recommended Usage

Use:

- `EasyOCR + Rules` for transparent rule-based parsing
- `LayoutLMv3 Only` for checkpoint evaluation and semantic extraction validation
- `Hybrid Extraction` for best-quality downstream receipt JSON
