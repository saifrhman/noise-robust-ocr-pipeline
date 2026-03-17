from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.receipt_ai.config import ReceiptAIConfig
from src.receipt_ai.dataset_loader import SROIEDatasetLoader
from src.receipt_ai.model.compatibility import inspect_checkpoint_label_space
from src.receipt_ai.model.preprocessing import load_layoutlmv3_processor
from src.receipt_ai.model.weak_label_analysis import analyze_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate training readiness before a long LayoutLMv3 run.")
    parser.add_argument("--dataset-root", default="data/SROIE2019")
    parser.add_argument("--train-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-json", default="outputs/training_sanity_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReceiptAIConfig.from_env()
    cfg.paths.data_root = Path(args.dataset_root).expanduser().resolve()

    loader = SROIEDatasetLoader(cfg.paths.data_root)
    processor = load_layoutlmv3_processor(args.model_name)

    train_summary, _ = analyze_split(
        loader,
        split=args.train_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=args.strict,
        max_samples=args.max_samples,
        item_name_threshold=30,
    )
    val_summary, _ = analyze_split(
        loader,
        split=args.val_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=args.strict,
        max_samples=args.max_samples,
        item_name_threshold=30,
    )

    train_truncation = _truncation_stats(
        loader,
        processor,
        split=args.train_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=args.strict,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    val_truncation = _truncation_stats(
        loader,
        processor,
        split=args.val_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=args.strict,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    compatibility = inspect_checkpoint_label_space(args.model_name)

    report: dict[str, Any] = {
        "dataset_root": str(cfg.paths.data_root),
        "available_splits": loader.available_splits(),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train": {
            "samples": train_summary["samples_analyzed"],
            "label_counts": train_summary["label_counts"],
            "token_distribution": train_summary["token_distribution"],
            "pseudo_only_samples": train_summary["pseudo_only_samples"],
            "truncation": train_truncation,
        },
        "val": {
            "samples": val_summary["samples_analyzed"],
            "label_counts": val_summary["label_counts"],
            "token_distribution": val_summary["token_distribution"],
            "pseudo_only_samples": val_summary["pseudo_only_samples"],
            "truncation": val_truncation,
        },
        "checkpoint_compatibility": {
            "target": args.model_name,
            "is_compatible": compatibility.is_compatible,
            "is_legacy": compatibility.is_legacy,
            "missing_entities": compatibility.missing_entities,
            "message": compatibility.message,
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training sanity check complete")
    print(f"Dataset root: {cfg.paths.data_root}")
    print(f"Train samples: {report['train']['samples']} | Val samples: {report['val']['samples']}")
    print(
        "Train tokens O/non-O: "
        f"{report['train']['token_distribution']['o_ratio']:.3f}/"
        f"{report['train']['token_distribution']['non_o_ratio']:.3f}"
    )
    print(
        "Train truncation: "
        f"{report['train']['truncation']['truncated_examples']} of "
        f"{report['train']['truncation']['examples_checked']} examples"
    )
    print(f"Checkpoint compatibility: {compatibility.message}")
    print(f"Saved report: {output_path}")


def _truncation_stats(
    loader: SROIEDatasetLoader,
    processor: Any,
    *,
    split: str,
    val_ratio: float,
    seed: int,
    strict: bool,
    max_length: int,
    max_samples: int,
) -> dict[str, Any]:
    examples_checked = 0
    truncated_examples = 0
    total_tokens_lost = 0
    max_word_count = 0

    for sample in loader.iter_samples(split, val_ratio=val_ratio, seed=seed, strict=strict):
        if max_samples > 0 and examples_checked >= max_samples:
            break
        words = [token.text for token in sample.ocr_tokens if token.text]
        boxes = [token.bbox.to_layoutlm_1000(sample.image_width, sample.image_height) for token in sample.ocr_tokens if token.text]
        if not words:
            continue

        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
            try:
                encoded = processor(
                    text=words,
                    images=image,
                    boxes=boxes,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
            except (TypeError, KeyError):
                encoded = processor(
                    words=words,
                    images=image,
                    boxes=boxes,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

        word_ids = encoded.word_ids(batch_index=0)
        encoded_words = len({word_id for word_id in word_ids if word_id is not None})
        lost = max(0, len(words) - encoded_words)

        max_word_count = max(max_word_count, len(words))
        total_tokens_lost += lost
        if lost > 0:
            truncated_examples += 1
        examples_checked += 1

    return {
        "examples_checked": examples_checked,
        "max_word_count_before_truncation": max_word_count,
        "truncated_examples": truncated_examples,
        "truncation_ratio": float(truncated_examples / examples_checked) if examples_checked else 0.0,
        "total_tokens_lost": total_tokens_lost,
        "avg_tokens_lost_per_example": float(total_tokens_lost / examples_checked) if examples_checked else 0.0,
    }


if __name__ == "__main__":
    main()
