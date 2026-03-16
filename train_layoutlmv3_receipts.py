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
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from data_utils import load_sroie_split_records, save_json, split_records_train_val
from label_config import ID2LABEL, LABEL2ID, SEMANTIC_LABELS, sanitize_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 for receipt key information extraction (SROIE-style)."
    )
    parser.add_argument("--train-dir", required=True, help="Split folder, for example data/sroie_kie/train")
    parser.add_argument("--val-dir", default=None, help="Optional validation split folder")
    parser.add_argument("--output-dir", required=True, help="Output checkpoint directory")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base", help="Base checkpoint")
    parser.add_argument("--image-dir-name", default="img", help="Image folder name inside split")
    parser.add_argument("--ocr-dir-name", default="box", help="OCR text folder name inside split")
    parser.add_argument(
        "--entity-dir-names",
        default="entities,key,keys,entity",
        help="Comma-separated annotation folder candidates",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="Used when --val-dir is not provided")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--ocr-mode", default="none", choices=["none", "clahe", "denoise", "otsu", "adaptive"])
    parser.add_argument(
        "--use-easyocr-when-missing-ocr",
        action="store_true",
        help="Fallback to EasyOCR if split does not include OCR txt files.",
    )
    return parser.parse_args()


def _load_processor(model_name: str) -> Any:
    try:
        return AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    except TypeError:
        return AutoProcessor.from_pretrained(model_name)


def _encode_with_processor(
    processor: Any,
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    labels: list[int],
) -> Any:
    common_kwargs = {
        "images": image,
        "boxes": boxes,
        "word_labels": labels,
        "truncation": True,
        "padding": "max_length",
        "return_tensors": "pt",
    }
    try:
        return processor(text=words, **common_kwargs)
    except (TypeError, KeyError):
        return processor(words=words, **common_kwargs)


def _make_encoder(processor: Any):
    def encode(example: dict[str, Any]) -> dict[str, Any]:
        image = Image.open(example["image_path"]).convert("RGB")
        labels = [LABEL2ID[label] for label in sanitize_labels(example["labels"])]

        encoded = _encode_with_processor(
            processor=processor,
            image=image,
            words=example["tokens"],
            boxes=example["bboxes"],
            labels=labels,
        )

        out: dict[str, Any] = {}
        for key, value in encoded.items():
            out[key] = value.squeeze(0).numpy()
        return out

    return encode


def _compute_metrics_factory(seqeval_metric: Any):
    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)

        pred_tags_all: list[list[str]] = []
        label_tags_all: list[list[str]] = []

        for pred_row, label_row in zip(preds, labels_np):
            pred_tags: list[str] = []
            label_tags: list[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if int(label_id) == -100:
                    continue
                pred_tags.append(ID2LABEL.get(int(pred_id), "O"))
                label_tags.append(ID2LABEL.get(int(label_id), "O"))
            pred_tags_all.append(pred_tags)
            label_tags_all.append(label_tags)

        metrics = seqeval_metric.compute(predictions=pred_tags_all, references=label_tags_all)
        return {
            "precision": float(metrics.get("overall_precision", 0.0)),
            "recall": float(metrics.get("overall_recall", 0.0)),
            "f1": float(metrics.get("overall_f1", 0.0)),
            "accuracy": float(metrics.get("overall_accuracy", 0.0)),
        }

    return compute_metrics


def _limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def _load_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve() if args.val_dir else None
    entity_dirs = [item.strip() for item in args.entity_dir_names.split(",") if item.strip()]

    train_rows = load_sroie_split_records(
        split_dir=train_dir,
        image_dir_name=args.image_dir_name,
        ocr_dir_name=args.ocr_dir_name,
        entity_dir_names=entity_dirs,
        max_words=args.max_words,
        use_easyocr_when_missing_ocr=args.use_easyocr_when_missing_ocr,
        ocr_mode=args.ocr_mode,
    )

    if val_dir:
        val_rows = load_sroie_split_records(
            split_dir=val_dir,
            image_dir_name=args.image_dir_name,
            ocr_dir_name=args.ocr_dir_name,
            entity_dir_names=entity_dirs,
            max_words=args.max_words,
            use_easyocr_when_missing_ocr=args.use_easyocr_when_missing_ocr,
            ocr_mode=args.ocr_mode,
        )
    else:
        train_rows, val_rows = split_records_train_val(train_rows, validation_ratio=args.validation_ratio, seed=args.seed)

    train_rows = _limit_rows(train_rows, args.max_train_samples)
    val_rows = _limit_rows(val_rows, args.max_val_samples)

    if not train_rows:
        raise ValueError("No training rows found. Check train split directories and entity annotations.")
    if not val_rows:
        raise ValueError("No validation rows found. Provide --val-dir or adjust --validation-ratio.")

    return train_rows, val_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = _load_records(args)

    processor = _load_processor(args.model_name)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(SEMANTIC_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    encode = _make_encoder(processor)
    train_ds = train_ds.map(encode, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(encode, remove_columns=val_ds.column_names)

    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    seqeval_metric = evaluate.load("seqeval")

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
        compute_metrics=_compute_metrics_factory(seqeval_metric),
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    save_json(output_dir / "label_mappings.json", {"label2id": LABEL2ID, "id2label": ID2LABEL})
    save_json(output_dir / "eval_metrics.json", metrics)

    run_config = {
        "model_name": args.model_name,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "args": vars(args),
    }
    save_json(output_dir / "run_config.json", run_config)

    print(json.dumps({"output_dir": str(output_dir), "eval_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
