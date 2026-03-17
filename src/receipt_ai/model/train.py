from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, set_seed

try:
    from transformers import EarlyStoppingCallback
except ImportError:  # pragma: no cover
    EarlyStoppingCallback = None

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.model.dataset import LayoutLMv3DatasetBuilder
from src.receipt_ai.model.labels import ID2LABEL, LABEL2ID, SEMANTIC_BIO_LABELS
from src.receipt_ai.model.metrics import compute_token_classification_metrics
from src.receipt_ai.model.preprocessing import load_layoutlmv3_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 token classification for receipt extraction.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--output-dir", default="outputs/layoutlmv3_receipt_ai")
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--eval-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.experiment_name:
        output_dir = output_dir / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_compatibility = inspect_checkpoint_label_space(args.model_name)
    if base_compatibility.normalized_labels and not base_compatibility.is_compatible:
        raise ValueError(
            "Refusing to train from a receipt checkpoint with an incompatible label space. "
            f"{base_compatibility.message}"
        )

    processor = load_layoutlmv3_processor(args.model_name)
    builder = LayoutLMv3DatasetBuilder(cfg)
    datasets = builder.build_train_val_datasets(
        processor,
        train_split=args.train_split,
        val_split=args.val_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_train=args.max_train_samples,
        limit_val=args.max_val_samples,
        strict=args.strict,
        max_length=args.max_seq_length,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(SEMANTIC_BIO_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
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
        metrics = compute_token_classification_metrics(label_tags_all, pred_tags_all)
        return {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "accuracy": float(metrics.get("token_accuracy", 0.0)),
        }

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "save_strategy": args.save_strategy,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "remove_unused_columns": False,
        "fp16": torch.cuda.is_available(),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "run_name": args.experiment_name or output_dir.name,
        "report_to": "none",
    }
    training_sig = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_sig:
        training_kwargs["evaluation_strategy"] = args.eval_strategy
    elif "eval_strategy" in training_sig:
        training_kwargs["eval_strategy"] = args.eval_strategy

    training_args = TrainingArguments(**training_kwargs)

    callbacks = []
    if args.early_stopping_patience > 0:
        if EarlyStoppingCallback is None:
            raise ImportError("Current transformers version does not expose EarlyStoppingCallback.")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.val_dataset,
        tokenizer=processor,
        data_collator=builder.data_collator(),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    train_metrics = trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    artifact = {
        "config": vars(args),
        "label_count": len(SEMANTIC_BIO_LABELS),
        "train_samples": len(datasets.train_records),
        "val_samples": len(datasets.val_records),
        "train_label_distribution": _label_distribution(datasets.train_records),
        "val_label_distribution": _label_distribution(datasets.val_records),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    (output_dir / "receipt_ai_training_summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def _label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.get("labels", []))
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
