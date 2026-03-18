from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.alignment import build_weak_bio_labels
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.model.inference import LayoutLMv3TokenClassifier
from src.receipt_ai.model.metrics import compute_token_classification_metrics
from src.receipt_ai.model.labels import sanitize_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LayoutLMv3 receipt token classification and save artifacts.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="outputs/layoutlmv3_sroie")
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default="outputs/layoutlmv3_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    backend = LayoutLMv3TokenClassifier(args.checkpoint)
    compatibility = inspect_checkpoint_label_space(args.checkpoint)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    predictions: list[dict[str, Any]] = []
    warnings_counter: Counter[str] = Counter()
    dropped_by_weak_label = 0
    for idx, sample in enumerate(_iter_samples(loader, args)):
        if idx >= args.max_samples:
            break
        weak = build_weak_bio_labels(sample)
        if weak.drop_training_sample:
            dropped_by_weak_label += 1
        gold_labels = sanitize_labels(weak.labels)
        pred = backend.predict(sample.image_path, sample.ocr_lines, sample.ocr_tokens, sample.image_width, sample.image_height)
        aligned_length = min(len(gold_labels), len(pred.token_labels), len(sample.ocr_tokens))
        gold_trimmed = gold_labels[:aligned_length]
        pred_trimmed = pred.token_labels[:aligned_length]
        gold_sequences.append(gold_trimmed)
        pred_sequences.append(pred_trimmed)

        for warning in pred.warnings:
            warnings_counter[warning] += 1

        result = pred.result.to_dict() if pred.result is not None else {}
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "assumptions": weak.assumptions,
                "sample_quality": weak.sample_quality,
                "filtering_summary": weak.filtering_summary,
                "drop_training_sample": weak.drop_training_sample,
                "drop_reason": weak.drop_reason,
                "warnings": pred.warnings,
                "tokens": [token.text for token in sample.ocr_tokens[:aligned_length]],
                "gold_labels": gold_trimmed,
                "predicted_labels": pred_trimmed,
                "predicted_scores": [float(score) for score in pred.token_scores[:aligned_length]],
                "prediction": result,
                "ground_truth": sample.ground_truth.to_dict() if sample.ground_truth is not None else None,
            }
        )

    metrics = compute_token_classification_metrics(gold_sequences, pred_sequences)
    summary: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "split": args.split,
        "samples_evaluated": len(predictions),
        "compatibility": {
            "is_compatible": compatibility.is_compatible,
            "is_legacy": compatibility.is_legacy,
            "missing_entities": compatibility.missing_entities,
            "message": compatibility.message,
        },
        "metrics": metrics,
        "warning_counts": dict(sorted(warnings_counter.items())),
        "weak_label_quality": {
            "samples_flagged_as_droppable": dropped_by_weak_label,
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Saved evaluation artifacts to {output_dir}")


def _iter_samples(loader: SROIEDatasetLoader, args: argparse.Namespace):
    return loader.iter_samples(args.split, val_ratio=args.val_ratio, seed=args.seed, strict=args.strict)


if __name__ == "__main__":
    main()
