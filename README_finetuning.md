# LayoutLMv3 Receipt Fine-Tuning Guide

## Why The Previous Approach Fails

Using `microsoft/layoutlmv3-base` directly for extraction fails for receipt KIE because it is a pretrained backbone, not a receipt-specific token-classification model.
Without task-specific fine-tuning, predictions typically use generic labels like `LABEL_0` and `LABEL_1`, which do not map reliably to fields such as merchant, date, address, or total.

## Label Schema

This project now uses an explicit semantic BIO schema from `label_config.py`:

- `O`
- `B-COMPANY`, `I-COMPANY`
- `B-DATE`, `I-DATE`
- `B-ADDRESS`, `I-ADDRESS`
- `B-TOTAL`, `I-TOTAL`

## Expected SROIE-Style Data Layout

```text
data/sroie_kie/
  train/
    img/
      X00016469670.jpg
      ...
    box/
      X00016469670.txt
      ...
    key/
      X00016469670.json
      ...
  val/  # optional if you want explicit validation split
    img/
    box/
    key/
```

Notes:
- `box/*.txt` uses OCR lines in SROIE format: `x1,y1,x2,y2,x3,y3,x4,y4,text`.
- `key/*.json` or `key/*.txt` should contain semantic field values (company/date/address/total).
- If `box/*.txt` is missing, you can enable EasyOCR fallback during training.

## Training Command (SROIE)

```bash
python train_layoutlmv3_receipts.py \
  --train-dir data/sroie_kie/train \
  --val-dir data/sroie_kie/val \
  --output-dir outputs/layoutlmv3_sroie \
  --model-name microsoft/layoutlmv3-base \
  --epochs 8 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --learning-rate 3e-5
```

Single split with automatic train/val split:

```bash
python train_layoutlmv3_receipts.py \
  --train-dir data/sroie_kie/train \
  --output-dir outputs/layoutlmv3_sroie \
  --validation-ratio 0.1
```

## Inference Command

```bash
python infer_layoutlmv3_receipt.py \
  --image data/sroie_kie/val/img/X00016469670.jpg \
  --checkpoint outputs/layoutlmv3_sroie \
  --pretty
```

Optional explicit OCR file:

```bash
python infer_layoutlmv3_receipt.py \
  --image data/sroie_kie/val/img/X00016469670.jpg \
  --ocr-path data/sroie_kie/val/box/X00016469670.txt \
  --checkpoint outputs/layoutlmv3_sroie \
  --pretty
```

## Sample Inference Output

```json
{
  "merchant": "ACME STORE SDN BHD",
  "date": "2018/01/25",
  "address": "NO 12 JALAN MAJU 1 KUALA LUMPUR",
  "total": "58.90",
  "raw_entities": [
    {
      "label": "COMPANY",
      "text": "ACME STORE SDN BHD",
      "score": 0.96,
      "bbox": [41, 22, 485, 70]
    }
  ]
}
```

## What Gets Saved After Training

In `--output-dir`:
- model checkpoints
- processor config
- `label_mappings.json`
- `eval_metrics.json`
- `run_config.json`
