# Noise-Robust OCR Pipeline

Receipt extraction system built around EasyOCR, a normalized rule parser, LayoutLMv3 token classification, and a hybrid fusion pipeline under [`src/receipt_ai`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai).

The repo now has one production pipeline:
- OCR: [`src/receipt_ai/ocr/easyocr_engine.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/ocr/easyocr_engine.py)
- Rules: [`src/receipt_ai/parsing/rules_parser.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/parsing/rules_parser.py)
- Model inference: [`src/receipt_ai/model/inference.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/model/inference.py)
- Fusion entrypoints: [`src/receipt_ai/pipelines/entrypoints.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/pipelines/entrypoints.py)
- Runtime defaults: [`default_config.json`](/home/saif/Projects/noise-robust-ocr-pipeline/default_config.json)

## Project Overview

The problem is noisy receipt extraction: OCR is imperfect, layout varies, and weak labels are often noisy. The solution in this repo is a practical layered pipeline:

- EasyOCR provides text and boxes.
- A rule parser extracts transparent baseline fields.
- LayoutLMv3 predicts semantic entities from OCR tokens and geometry.
- Hybrid fusion decides when model evidence should replace or supplement rules.
- Training, evaluation, comparison, ablation, and experiment diagnosis scripts keep the system evidence-driven.

## Architecture

Runtime flow:

1. Image input
2. OCR tokenization and line grouping
3. One of:
   - rules only
   - model only
   - hybrid fusion
4. Schema-normalized JSON output

Main runtime files:
- [`run_receipt_ai.py`](/home/saif/Projects/noise-robust-ocr-pipeline/run_receipt_ai.py): simple CLI entrypoint
- [`app.py`](/home/saif/Projects/noise-robust-ocr-pipeline/app.py): Streamlit demo app
- [`src/receipt_ai/runtime/policy.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/runtime/policy.py): default config loading
- [`src/receipt_ai/runtime/runner.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/runtime/runner.py): shared mode resolution and deterministic runtime setup
- [`src/receipt_ai/runtime/output.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/runtime/output.py): standardized JSON shaping

## Modes

`easyocr_rules`
- Most stable fallback.
- Uses OCR plus rule parsing only.
- Best choice when checkpoint evidence is weak or no valid checkpoint is available.

`layoutlm_only`
- Pure model extraction.
- Useful for evaluating checkpoint quality directly.
- Requires a valid fine-tuned receipt checkpoint.

`hybrid`
- Runs rules and model, then fuses fields by confidence and semantic thresholds.
- Intended production mode when model evidence is strong enough.

The active repo default comes from [`default_config.json`](/home/saif/Projects/noise-robust-ocr-pipeline/default_config.json). CLI and Streamlit both use it automatically unless overridden.

## Output Schema

Final JSON always follows the normalized schema from [`src/receipt_ai/schemas.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/receipt_ai/schemas.py):

```json
{
  "vendor": {
    "name": "",
    "registration_number": "",
    "address": ""
  },
  "invoice": {
    "invoice_type": "",
    "bill_number": "",
    "order_number": "",
    "table_number": "",
    "date": "",
    "time": "",
    "cashier": ""
  },
  "items": [
    {
      "name": "",
      "quantity": 1.0,
      "unit_price": 0.0,
      "line_total": 0.0
    }
  ],
  "totals": {
    "subtotal": 0.0,
    "service_charge": 0.0,
    "tax": 0.0,
    "rounding": 0.0,
    "total": 0.0,
    "currency": ""
  },
  "payment": {
    "method": "",
    "amount_paid": 0.0
  },
  "metadata": {
    "mode": "easyocr_rules",
    "source_image": "receipt.jpg",
    "warnings": [],
    "confidence": 0.0,
    "field_confidences": {},
    "field_provenance": {}
  }
}
```

Output modes:
- `full`: includes confidence and provenance when enabled
- `minimal`: keeps the same top-level schema but strips confidence/provenance details

## Quickstart

Install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required. This repo uses `dataclass(slots=True)` and should be run with `python3.10+` or `python3.11`.

Run one receipt with the default config:

```bash
python3.11 run_receipt_ai.py path/to/receipt.jpg
```

Save JSON to a file:

```bash
python3.11 run_receipt_ai.py path/to/receipt.jpg --output outputs/demo_receipt.json
```

Run a folder:

```bash
python3.11 run_receipt_ai.py path/to/receipt_folder --output outputs/demo_batch
```

## Streamlit App

Run:

```bash
streamlit run app.py
```

The app:
- loads [`default_config.json`](/home/saif/Projects/noise-robust-ocr-pipeline/default_config.json) automatically
- shows the selected mode, effective mode, checkpoint used, and fallback behavior
- allows manual override of mode, checkpoint, and output verbosity

## Training

Short training run:

```bash
python3.11 scripts/train_receipt_ai_layoutlmv3.py \
  --dataset-root data/SROIE2019 \
  --model-name microsoft/layoutlmv3-base \
  --output-dir outputs/layoutlmv3_receipt_ai \
  --experiment-name smoke_richer_run \
  --smoke \
  --loss-type focal \
  --focal-gamma 2.0 \
  --use-class-weights \
  --oversample-non-o \
  --drop-noisy-samples \
  --critical-label-boost 1.75 \
  --weak-label-floor 0.40
```

## Experiment Cycle

Full baseline vs improved experiment:

```bash
python3.11 scripts/run_receipt_ai_experiment_cycle.py \
  --baseline-checkpoint outputs/layoutlmv3_sroie \
  --base-model microsoft/layoutlmv3-base \
  --dataset-root data/SROIE2019 \
  --experiment-name improved_vs_baseline_val \
  --output-dir outputs/experiments \
  --split val \
  --seed 42 \
  --max-train-samples 128 \
  --max-eval-samples 64 \
  --epochs 1
```

Postmortem on an experiment folder:

```bash
python3.11 scripts/run_experiment_postmortem_receipt_ai.py \
  --experiment-root outputs/experiments/improved_vs_baseline_val
```

Finalize repo default config after reviewing artifacts:

```bash
python3.11 scripts/finalize_default_config_receipt_ai.py
```

## Demo Flow

Minimal local demo:

```bash
python3.11 run_receipt_ai.py outputs/tmp_test/dummy.jpg --output-mode minimal
```

If you want the same image in Streamlit:

```bash
streamlit run app.py
```

## Legacy Modules

These are retained only as deprecated compatibility paths and should not be used for new work:
- [`src/app/extraction_modes.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/app/extraction_modes.py)
- [`src/app/extract_fields.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/app/extract_fields.py)
- [`src/app/text_cleaning.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/app/text_cleaning.py)
- [`src/app/receipt_script_parser.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/app/receipt_script_parser.py)
- [`src/layoutlmv3_engine.py`](/home/saif/Projects/noise-robust-ocr-pipeline/src/layoutlmv3_engine.py)
- [`infer_layoutlmv3_receipt.py`](/home/saif/Projects/noise-robust-ocr-pipeline/infer_layoutlmv3_receipt.py)

## Limitations

- Weak labels remain a bottleneck for model quality.
- SROIE supervision is limited and does not cover all richer-schema fields equally well.
- Item coherence analysis is heuristic rather than gold-labeled.
- Hybrid quality depends on checkpoint quality and threshold tuning.
- Default mode may remain rules-first when experiment evidence is too weak to promote a model-heavy default.

## Future Work

- Improve weak-label precision on vendor/date/total and item spans.
- Expand evaluation beyond SROIE-style supervision.
- Strengthen item decoding and fusion decisions with more labeled evidence.
- Promote a stronger checkpoint once experiment evidence is large enough and stable enough to justify it.
