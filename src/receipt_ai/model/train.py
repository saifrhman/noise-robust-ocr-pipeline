from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler, WeightedRandomSampler
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments, set_seed

try:
    from transformers import EarlyStoppingCallback
except ImportError:  # pragma: no cover
    EarlyStoppingCallback = None

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.model.dataset import LayoutLMv3DatasetBuilder
from src.receipt_ai.model.labels import CRITICAL_ENTITY_NAMES, ID2LABEL, LABEL2ID, SEMANTIC_BIO_LABELS, entity_from_label
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
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--oversample-non-o", action="store_true")
    parser.add_argument("--drop-noisy-samples", action="store_true")
    parser.add_argument("--weak-label-floor", type=float, default=0.35)
    parser.add_argument("--critical-label-boost", type=float, default=1.5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


class ImbalanceAwareTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor | None = None,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        sample_weights: list[float] | None = None,
        weak_label_floor: float = 0.35,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = max(0.0, float(label_smoothing))
        self.sample_weights = sample_weights
        self.weak_label_floor = max(0.0, float(weak_label_floor))

    def _get_train_sampler(self, train_dataset=None) -> Sampler[Any] | None:
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not self.sample_weights or dataset is None or not hasattr(dataset, "__len__"):
            return super()._get_train_sampler(train_dataset=train_dataset)
        if len(self.sample_weights) != len(dataset):
            return super()._get_train_sampler(train_dataset=train_dataset)
        return WeightedRandomSampler(weights=self.sample_weights, num_samples=len(self.sample_weights), replacement=True)

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ):
        labels = inputs.get("labels")
        label_weights = inputs.get("label_weights")
        outputs = model(**{k: v for k, v in inputs.items() if k not in {"labels", "label_weights"}})
        logits = outputs.get("logits")

        if labels is None or logits is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
                raise ValueError("Model outputs do not include loss/logits and labels were not provided.")
            return (loss, outputs) if return_outputs else loss

        active_mask = labels != -100
        active_logits = logits[active_mask]
        active_labels = labels[active_mask]
        active_weights = label_weights[active_mask] if label_weights is not None else None
        if active_logits.numel() == 0:
            zero = logits.sum() * 0.0
            return (zero, outputs) if return_outputs else zero

        weight = self.class_weights.to(active_logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(
            active_logits,
            active_labels,
            weight=weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.loss_type == "focal":
            pt = torch.exp(-ce)
            loss_terms = ((1.0 - pt) ** self.focal_gamma) * ce
        else:
            loss_terms = ce

        if active_weights is not None:
            scaled = torch.clamp(active_weights.to(loss_terms.device), min=self.weak_label_floor)
            loss_terms = loss_terms * scaled
            denom = torch.clamp(scaled.sum(), min=1e-8)
            loss = loss_terms.sum() / denom
        else:
            loss = loss_terms.mean()
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 64
        args.max_val_samples = args.max_val_samples or 32
        args.save_strategy = "epoch"
        args.eval_strategy = "epoch"
        args.logging_steps = min(args.logging_steps, 10)
        args.experiment_name = args.experiment_name or "smoke"
        args.drop_noisy_samples = True

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
        drop_noisy_samples=args.drop_noisy_samples,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(SEMANTIC_BIO_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    latest_eval_details: dict[str, Any] = {}

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
        latest_eval_details.clear()
        latest_eval_details.update(metrics)
        return {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "accuracy": float(metrics.get("token_accuracy", 0.0)),
            "vendor_f1": float(((metrics.get("critical_fields", {}) or {}).get("VENDOR_NAME", {}) or {}).get("f1", 0.0)),
            "date_f1": float(((metrics.get("critical_fields", {}) or {}).get("DATE", {}) or {}).get("f1", 0.0)),
            "total_f1": float(((metrics.get("critical_fields", {}) or {}).get("TOTAL", {}) or {}).get("f1", 0.0)),
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

    callbacks: list[Any] = []
    if args.early_stopping_patience > 0:
        if EarlyStoppingCallback is None:
            raise ImportError("Current transformers version does not expose EarlyStoppingCallback.")
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    class_weights = (
        _build_class_weights(datasets.train_records, critical_label_boost=args.critical_label_boost)
        if args.use_class_weights
        else None
    )
    sample_weights = _build_sample_weights(datasets.train_records) if args.oversample_non_o else None

    trainer = ImbalanceAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.val_dataset,
        tokenizer=processor,
        data_collator=builder.data_collator(),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        sample_weights=sample_weights,
        weak_label_floor=args.weak_label_floor,
    )
    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    artifact = {
        "config": vars(args),
        "label_count": len(SEMANTIC_BIO_LABELS),
        "train_samples": len(datasets.train_records),
        "val_samples": len(datasets.val_records),
        "imbalance_strategy": {
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "use_class_weights": args.use_class_weights,
            "oversample_non_o": args.oversample_non_o,
            "weak_label_floor": args.weak_label_floor,
            "critical_label_boost": args.critical_label_boost,
            "drop_noisy_samples": args.drop_noisy_samples,
        },
        "train_label_distribution": _label_distribution(datasets.train_records),
        "val_label_distribution": _label_distribution(datasets.val_records),
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "dataset_diagnostics": datasets.diagnostics,
        "eval_detailed_metrics": latest_eval_details,
        "class_weights": _tensor_to_dict(class_weights),
        "sample_weight_summary": _sample_weight_summary(sample_weights),
    }
    (output_dir / "receipt_ai_training_summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (output_dir / "receipt_ai_training_diagnostics.json").write_text(
        json.dumps(_build_training_diagnostics(datasets.train_records, latest_eval_details), indent=2),
        encoding="utf-8",
    )


def _label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.get("labels", []))
    return dict(sorted(counts.items()))


def _build_class_weights(records: list[dict[str, Any]], *, critical_label_boost: float) -> torch.Tensor:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.get("labels", []))

    weights = torch.ones(len(LABEL2ID), dtype=torch.float)
    total = sum(max(count, 1) for count in counts.values())
    for label, idx in LABEL2ID.items():
        count = max(int(counts.get(label, 0)), 1)
        weights[idx] = float(total / (len(LABEL2ID) * count))
        if entity_from_label(label) in CRITICAL_ENTITY_NAMES and label != "O":
            weights[idx] *= max(1.0, float(critical_label_boost))
    weights = weights / max(weights.mean().item(), 1e-8)
    return weights


def _build_sample_weights(records: list[dict[str, Any]]) -> list[float]:
    weights: list[float] = []
    for record in records:
        labels = list(record.get("labels", []))
        if not labels:
            weights.append(1.0)
            continue
        non_o = sum(1 for label in labels if label != "O")
        ratio = non_o / max(len(labels), 1)
        critical_bonus = 0.0
        label_set = set(labels)
        for entity in CRITICAL_ENTITY_NAMES:
            if any(label.endswith(f"-{entity}") for label in label_set):
                critical_bonus += 0.25
        weights.append(max(0.25, 0.5 + 5.0 * ratio + critical_bonus))
    return weights


def _build_training_diagnostics(records: list[dict[str, Any]], eval_details: dict[str, Any]) -> dict[str, Any]:
    total_tokens = 0
    labeled_tokens = 0
    weight_sum = 0.0
    weight_count = 0
    per_label_weights: dict[str, list[float]] = {}
    filtering_summary: Counter[str] = Counter()
    for record in records:
        labels = list(record.get("labels", []))
        weights = list(record.get("label_weights", []))
        total_tokens += len(labels)
        for label, weight in zip(labels, weights):
            if label == "O":
                continue
            labeled_tokens += 1
            weight_sum += float(weight)
            weight_count += 1
            per_label_weights.setdefault(label, []).append(float(weight))
        filtering_summary.update({k: int(v) for k, v in (record.get("filtering_summary", {}) or {}).items()})

    avg_label_weight = {
        label: float(sum(values) / len(values)) for label, values in sorted(per_label_weights.items()) if values
    }
    return {
        "label_distribution_after_filtering": _label_distribution(records),
        "token_distribution_after_filtering": {
            "total_tokens": total_tokens,
            "labeled_tokens": labeled_tokens,
            "labeled_ratio": float(labeled_tokens / total_tokens) if total_tokens else 0.0,
            "o_ratio": float((total_tokens - labeled_tokens) / total_tokens) if total_tokens else 0.0,
        },
        "avg_non_o_weight": float(weight_sum / weight_count) if weight_count else 0.0,
        "avg_weight_per_label": avg_label_weight,
        "filtering_summary": dict(sorted(filtering_summary.items())),
        "per_label_prediction_frequency": dict(sorted((eval_details.get("prediction_frequency", {}) or {}).items())),
        "critical_field_metrics": eval_details.get("critical_fields", {}),
    }


def _tensor_to_dict(weights: torch.Tensor | None) -> dict[str, float]:
    if weights is None:
        return {}
    return {ID2LABEL[idx]: float(weights[idx].item()) for idx in range(len(weights))}


def _sample_weight_summary(sample_weights: list[float] | None) -> dict[str, float]:
    if not sample_weights:
        return {}
    values = [float(weight) for weight in sample_weights]
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


if __name__ == "__main__":
    main()
