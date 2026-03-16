from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


REQUIRED_KEYS = {"image_path", "tokens", "bboxes", "labels"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 for receipt token classification."
    )
    parser.add_argument("--train-jsonl", required=True, help="Path to training JSONL")
    parser.add_argument("--val-jsonl", required=True, help="Path to validation JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base", help="Base model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)

            missing = REQUIRED_KEYS.difference(row.keys())
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(f"{path}:{line_no} missing keys: {missing_str}")

            tokens = row["tokens"]
            bboxes = row["bboxes"]
            labels = row["labels"]

            if not (len(tokens) == len(bboxes) == len(labels)):
                raise ValueError(
                    f"{path}:{line_no} has mismatched token/bbox/label lengths "
                    f"({len(tokens)}/{len(bboxes)}/{len(labels)})"
                )

            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()

            rows.append(
                {
                    "id": row.get("id", f"{path.stem}-{line_no}"),
                    "image_path": str(image_path),
                    "tokens": tokens,
                    "bboxes": bboxes,
                    "labels": labels,
                }
            )

            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No samples found in {path}")

    return rows


def build_label_set(rows: list[dict[str, Any]]) -> list[str]:
    labels = sorted({label for row in rows for label in row["labels"]})
    if "O" in labels:
        labels = ["O"] + [label for label in labels if label != "O"]
    return labels


def normalize_bboxes(bboxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    if not bboxes:
        return []

    width = max(width, 1)
    height = max(height, 1)

    out: list[list[int]] = []
    for box in bboxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        if max(x1, y1, x2, y2) <= 1000 and min(x1, y1, x2, y2) >= 0:
            out.append([x1, y1, x2, y2])
            continue

        out.append(
            [
                int(1000 * x1 / width),
                int(1000 * y1 / height),
                int(1000 * x2 / width),
                int(1000 * y2 / height),
            ]
        )

    return out


def encode_layout_features(processor, image: Image.Image, tokens: list[str], boxes: list[list[int]], word_labels: list[int]):
    """
    Handle LayoutLMv3 processor API differences across transformers versions.
    """
    common_kwargs = {
        "images": image,
        "boxes": boxes,
        "word_labels": word_labels,
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }

    try:
        return processor(text=tokens, **common_kwargs)
    except (TypeError, KeyError):
        return processor(words=tokens, **common_kwargs)


def make_encoder(processor, label2id: dict[str, int]):
    def encode(example: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(example["image_path"]).convert("RGB")
        word_labels = [label2id[label] for label in example["labels"]]
        boxes = normalize_bboxes(example["bboxes"], image.width, image.height)

        encoded = encode_layout_features(
            processor=processor,
            image=image,
            tokens=example["tokens"],
            boxes=boxes,
            word_labels=word_labels,
        )

        out: dict[str, Any] = {}
        for key, value in encoded.items():
            out[key] = value.squeeze(0).numpy()
        return out

    return encode


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_jsonl).resolve()
    val_path = Path(args.val_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(train_path, args.max_train_samples)
    val_rows = read_jsonl(val_path, args.max_val_samples)

    labels = build_label_set(train_rows + val_rows)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    try:
        processor = AutoProcessor.from_pretrained(args.model_name, apply_ocr=False)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    encode = make_encoder(processor, label2id)

    train_ds = train_ds.map(encode, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(encode, remove_columns=val_ds.column_names)

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []

        for pred_row, label_row in zip(preds, labels_np):
            pred_tags: list[str] = []
            label_tags: list[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                pred_tags.append(id2label[int(pred_id)])
                label_tags.append(id2label[int(label_id)])
            true_predictions.append(pred_tags)
            true_labels.append(label_tags)

        metrics = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": metrics["overall_precision"],
            "recall": metrics["overall_recall"],
            "f1": metrics["overall_f1"],
            "accuracy": metrics["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    metrics_path = output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved fine-tuned model to: {output_dir}")
    print(f"Saved eval metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
